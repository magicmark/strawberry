from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from functools import cmp_to_key, partial
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    cast,
)

import rich
from graphql import (
    BooleanValueNode,
    EnumValueNode,
    FieldNode,
    FloatValueNode,
    FragmentDefinitionNode,
    FragmentSpreadNode,
    GraphQLEnumType,
    GraphQLInputObjectType,
    GraphQLInterfaceType,
    GraphQLList as GQLList,
    GraphQLNonNull,
    GraphQLObjectType as GQLObjectType,
    GraphQLScalarType,
    GraphQLUnionType,
    InlineFragmentNode,
    IntValueNode,
    ListTypeNode,
    ListValueNode,
    NamedTypeNode,
    NonNullTypeNode,
    NullValueNode,
    ObjectValueNode,
    OperationDefinitionNode,
    StringValueNode,
    Undefined,
    VariableNode,
    build_schema,
    parse,
)
from typing_extensions import Protocol

from strawberry.types.unset import UNSET

try:
    from dataclasses import MISSING
except ImportError:
    MISSING = object()  # type: ignore[misc,assignment]

from strawberry.utils.str_converters import capitalize_first, to_camel_case

from .exceptions import (
    MultipleOperationsProvidedError,
    NoOperationNameProvidedError,
    NoOperationProvidedError,
)
from .types import (
    GraphQLArgument,
    GraphQLBoolValue,
    GraphQLDirective,
    GraphQLEnum,
    GraphQLEnumValue,
    GraphQLField,
    GraphQLFieldSelection,
    GraphQLFloatValue,
    GraphQLFragmentSpread,
    GraphQLFragmentType,
    GraphQLInlineFragment,
    GraphQLIntValue,
    GraphQLList,
    GraphQLListValue,
    GraphQLNullValue,
    GraphQLObjectType,
    GraphQLObjectValue,
    GraphQLOperation,
    GraphQLOptional,
    GraphQLScalar,
    GraphQLStringValue,
    GraphQLUnion,
    GraphQLVariable,
    GraphQLVariableReference,
)

if TYPE_CHECKING:
    from graphql import (
        ArgumentNode,
        DirectiveNode,
        DocumentNode,
        GraphQLOutputType,
        GraphQLSchema,
        SelectionNode,
        SelectionSetNode,
        TypeNode,
        ValueNode,
        VariableDefinitionNode,
    )

    from .types import GraphQLSelection, GraphQLType

from .types import GraphQLArgumentValue


@dataclass
class CodegenFile:
    path: str
    content: str


@dataclass
class CodegenResult:
    files: list[CodegenFile]

    def to_string(self) -> str:
        return "\n".join(f.content for f in self.files) + "\n"

    def write(self, folder: Path) -> None:
        for file in self.files:
            destination = folder / file.path
            destination.parent.mkdir(exist_ok=True, parents=True)
            destination.write_text(file.content)


class HasSelectionSet(Protocol):
    selection_set: SelectionSetNode | None


_TYPE_TO_GRAPHQL_TYPE: dict[type, type[GraphQLArgumentValue]] = {
    str: GraphQLStringValue,
    int: GraphQLIntValue,
    float: GraphQLFloatValue,
    bool: GraphQLBoolValue,
}


def _py_to_graphql_value(obj: Any) -> GraphQLArgumentValue:
    """Convert a python object to a GraphQLArgumentValue."""
    if obj is None or obj is UNSET:
        return GraphQLNullValue(value=obj)

    obj_type = type(obj)
    if obj_type in _TYPE_TO_GRAPHQL_TYPE:
        return _TYPE_TO_GRAPHQL_TYPE[obj_type](obj)
    if issubclass(obj_type, Enum):
        return GraphQLEnumValue(obj.name, enum_type=obj_type.__name__)
    if issubclass(obj_type, Sequence):
        return GraphQLListValue([_py_to_graphql_value(v) for v in obj])
    if issubclass(obj_type, Mapping):
        return GraphQLObjectValue({k: _py_to_graphql_value(v) for k, v in obj.items()})
    raise ValueError(f"Cannot convert {obj!r} into a GraphQLArgumentValue")


class QueryCodegenPlugin:
    def __init__(self, query: Path) -> None:
        """Initialize the plugin.

        The singular argument is the path to the file that is being processed
        by this plugin.
        """
        self.query = query

    def on_start(self) -> None: ...

    def on_end(self, result: CodegenResult) -> None: ...

    def generate_code(
        self, types: list[GraphQLType], operation: GraphQLOperation
    ) -> list[CodegenFile]:
        return []


class ConsolePlugin:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.files_generated: list[Path] = []

    def before_any_start(self) -> None:
        rich.print(
            "[bold yellow]The codegen is experimental. Please submit any bug at "
            "https://github.com/strawberry-graphql/strawberry\n",
        )

    def after_all_finished(self) -> None:
        rich.print("[green]Generated:")
        for fname in self.files_generated:
            rich.print(f"  {fname}")

    def on_start(self, plugins: Iterable[QueryCodegenPlugin], query: Path) -> None:
        plugin_names = [plugin.__class__.__name__ for plugin in plugins]

        rich.print(
            f"[green]Generating code for {query} using "
            f"{', '.join(plugin_names)} plugin(s)",
        )

    def on_end(self, result: CodegenResult) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        result.write(self.output_dir)

        self.files_generated.extend(Path(cf.path) for cf in result.files)

        rich.print(
            f"[green] Generated {len(result.files)} files in {self.output_dir}",
        )


def _get_deps(t: GraphQLType) -> Iterable[GraphQLType]:
    """Get all the types that `t` depends on.

    To keep things simple, `t` depends on itself.
    """
    yield t

    if isinstance(t, GraphQLObjectType):
        for fld in t.fields:
            yield from _get_deps(fld.type)

    elif isinstance(t, (GraphQLEnum, GraphQLScalar)):
        # enums and scalars have no dependent types
        pass

    elif isinstance(t, (GraphQLOptional, GraphQLList)):
        yield from _get_deps(t.of_type)

    elif isinstance(t, GraphQLUnion):
        for gql_type in t.types:
            yield from _get_deps(gql_type)
    else:
        # Want to make sure that all types are covered.
        raise ValueError(f"Unknown GraphQLType: {t}")  # noqa: TRY004


class QueryCodegenPluginManager:
    def __init__(
        self,
        plugins: list[QueryCodegenPlugin],
        console_plugin: ConsolePlugin | None = None,
    ) -> None:
        self.plugins = plugins
        self.console_plugin = console_plugin

    def _sort_types(self, types: list[GraphQLType]) -> list[GraphQLType]:
        """Sort the types.

        t1 < t2 iff t2 has a dependency on t1.
        t1 == t2 iff neither type has a dependency on the other.
        """

        def type_cmp(t1: GraphQLType, t2: GraphQLType) -> int:
            """Compare the types."""
            if t1 is t2:
                retval = 0
            elif t1 in _get_deps(t2):
                retval = -1
            elif t2 in _get_deps(t1):
                retval = 1
            else:
                retval = 0

            return retval

        return sorted(types, key=cmp_to_key(type_cmp))

    def generate_code(
        self, types: list[GraphQLType], operation: GraphQLOperation
    ) -> CodegenResult:
        result = CodegenResult(files=[])

        types = self._sort_types(types)

        for plugin in self.plugins:
            files = plugin.generate_code(types, operation)

            result.files.extend(files)

        return result

    def on_start(self) -> None:
        if self.console_plugin and self.plugins:
            # We need the query that we're processing
            # just pick it off the first plugin
            query = self.plugins[0].query
            self.console_plugin.on_start(self.plugins, query)

        for plugin in self.plugins:
            plugin.on_start()

    def on_end(self, result: CodegenResult) -> None:
        for plugin in self.plugins:
            plugin.on_end(result)

        if self.console_plugin:
            self.console_plugin.on_end(result)


class QueryCodegen:
    """Query codegen that works with graphql-core's GraphQLSchema.

    This class processes GraphQL queries against a schema and generates
    code using the configured plugins. It can work with:
    - SDL strings (via from_sdl class method)
    - Strawberry schemas (automatically detected and converted)
    - Raw graphql-core GraphQLSchema objects
    """

    def __init__(
        self,
        schema: GraphQLSchema,
        plugins: list[QueryCodegenPlugin],
        console_plugin: ConsolePlugin | None = None,
        *,
        scalar_registry: dict[str, type] | None = None,
        enum_registry: dict[str, type] | None = None,
    ) -> None:
        # Handle Strawberry Schema objects for backwards compatibility
        gql_schema, scalar_reg, enum_reg = self._extract_schema_info(schema)

        self.schema = gql_schema
        self.plugin_manager = QueryCodegenPluginManager(plugins, console_plugin)
        self.types: list[GraphQLType] = []
        # Registries map GraphQL type names to Python types
        # Use provided registries, or ones extracted from Strawberry schema
        self.scalar_registry = scalar_registry if scalar_registry is not None else scalar_reg
        self.enum_registry = enum_registry if enum_registry is not None else enum_reg

    @staticmethod
    def _extract_schema_info(
        schema: Any,
    ) -> tuple[GraphQLSchema, dict[str, type], dict[str, type]]:
        """Extract graphql-core schema and type registries from a schema.

        Handles both raw GraphQLSchema and Strawberry Schema objects.
        """
        from graphql import GraphQLSchema as GQLSchema

        # If it's already a graphql-core schema, return as-is
        if isinstance(schema, GQLSchema):
            return schema, {}, {}

        # Check if it's a Strawberry Schema by looking for _schema attribute
        if hasattr(schema, "_schema") and hasattr(schema, "schema_converter"):
            gql_schema = schema._schema
            scalar_registry = QueryCodegen._build_scalar_registry(schema)
            enum_registry = QueryCodegen._build_enum_registry(schema)
            return gql_schema, scalar_registry, enum_registry

        # Fallback: assume it has the interface we need
        raise TypeError(
            f"Expected GraphQLSchema or Strawberry Schema, got {type(schema).__name__}"
        )

    @staticmethod
    def _build_scalar_registry(schema: Any) -> dict[str, type]:
        """Build scalar registry from a Strawberry schema."""
        registry: dict[str, type] = {}

        def _get_base_type(origin: Any) -> type | None:
            """Get the base Python type from an origin, unwrapping NewType if needed."""
            if origin is None:
                return None
            # For NewType scalars, get the underlying type
            if hasattr(origin, "__supertype__"):
                return origin.__supertype__
            return origin

        # First, try getting from schema_converter.scalar_registry
        for _python_type, scalar_def in schema.schema_converter.scalar_registry.items():
            if hasattr(scalar_def, "name") and hasattr(scalar_def, "origin"):
                base_type = _get_base_type(scalar_def.origin)
                if base_type is not None:
                    registry[scalar_def.name] = base_type
        # Also check graphql-core type_map for scalars with extensions
        # (custom scalars may have origin in extensions, not scalar_registry)
        for type_name, gql_type in schema._schema.type_map.items():
            if isinstance(gql_type, GraphQLScalarType) and type_name not in registry:
                ext = getattr(gql_type, "extensions", {})
                strawberry_def = ext.get("strawberry-definition")
                if strawberry_def and hasattr(strawberry_def, "origin"):
                    base_type = _get_base_type(strawberry_def.origin)
                    if base_type is not None:
                        registry[type_name] = base_type
        return registry

    @staticmethod
    def _build_enum_registry(schema: Any) -> dict[str, type]:
        """Build enum registry from a Strawberry schema."""
        registry: dict[str, type] = {}
        for type_name, gql_type in schema._schema.type_map.items():
            if isinstance(gql_type, GraphQLEnumType):
                extensions = getattr(gql_type, "extensions", {})
                strawberry_def = extensions.get("strawberry-definition")
                if strawberry_def and hasattr(strawberry_def, "wrapped_cls"):
                    registry[type_name] = strawberry_def.wrapped_cls
        return registry

    def _get_field_name(self, field: Any, graphql_name: str) -> str:
        """Get the Python field name for codegen output.

        For Strawberry schemas, uses the Python name from the field definition.
        For SDL schemas, uses the GraphQL name as-is.
        """
        # Try to get the Strawberry field definition from extensions
        extensions = getattr(field, "extensions", {})
        strawberry_def = extensions.get("strawberry-definition")
        if strawberry_def is not None:
            # Use graphql_name if explicitly set, otherwise use the Python name
            if (
                hasattr(strawberry_def, "graphql_name")
                and strawberry_def.graphql_name is not None
            ):
                return strawberry_def.graphql_name
            if hasattr(strawberry_def, "name"):
                return strawberry_def.name
        # Fall back to GraphQL name for SDL schemas
        return graphql_name

    @classmethod
    def from_sdl(
        cls,
        sdl: str,
        plugins: list[QueryCodegenPlugin],
        console_plugin: ConsolePlugin | None = None,
    ) -> QueryCodegen:
        """Create a QueryCodegen from an SDL string."""
        schema = build_schema(sdl)
        return cls(schema, plugins, console_plugin)

    def run(self, query: str) -> CodegenResult:
        self.plugin_manager.on_start()

        ast = parse(query)

        operations = self._get_operations(ast)

        if not operations:
            raise NoOperationProvidedError

        if len(operations) > 1:
            raise MultipleOperationsProvidedError

        operation = operations[0]

        if operation.name is None:
            raise NoOperationNameProvidedError

        self._populate_fragment_types(ast)
        self.operation = self._convert_operation(operation)

        result = self.generate_code()
        self.plugin_manager.on_end(result)

        return result

    def _collect_type(self, type_: GraphQLType) -> None:
        if type_ in self.types:
            return
        self.types.append(type_)

    def _populate_fragment_types(self, ast: DocumentNode) -> None:
        fragment_definitions = (
            definition
            for definition in ast.definitions
            if isinstance(definition, FragmentDefinitionNode)
        )
        for fd in fragment_definitions:
            query_type = self._get_parent_type_by_name(fd.type_condition.name.value)

            typename = fd.type_condition.name.value
            graph_ql_object_type_factory = partial(
                GraphQLFragmentType,
                on=typename,
                graphql_typename=typename,
            )

            self._collect_types(
                cast("HasSelectionSet", fd),
                parent_type=query_type,
                class_name=fd.name.value,
                graph_ql_object_type_factory=graph_ql_object_type_factory,
            )

    def _convert_selection(self, selection: SelectionNode) -> GraphQLSelection:
        if isinstance(selection, FieldNode):
            return GraphQLFieldSelection(
                selection.name.value,
                selection.alias.value if selection.alias else None,
                selections=self._convert_selection_set(selection.selection_set),
                directives=self._convert_directives(selection.directives),
                arguments=self._convert_arguments(selection.arguments),
            )

        if isinstance(selection, InlineFragmentNode):
            return GraphQLInlineFragment(
                selection.type_condition.name.value,
                self._convert_selection_set(selection.selection_set),
            )

        if isinstance(selection, FragmentSpreadNode):
            return GraphQLFragmentSpread(selection.name.value)

        raise ValueError(f"Unsupported type: {type(selection)}")  # pragma: no cover

    def _convert_selection_set(
        self, selection_set: SelectionSetNode | None
    ) -> list[GraphQLSelection]:
        if selection_set is None:
            return []
        return [
            self._convert_selection(selection) for selection in selection_set.selections
        ]

    def _convert_value(self, value: ValueNode) -> GraphQLArgumentValue:
        if isinstance(value, StringValueNode):
            return GraphQLStringValue(value.value)
        if isinstance(value, IntValueNode):
            return GraphQLIntValue(int(value.value))
        if isinstance(value, FloatValueNode):
            return GraphQLFloatValue(float(value.value))
        if isinstance(value, NullValueNode):
            return GraphQLNullValue()
        if isinstance(value, VariableNode):
            return GraphQLVariableReference(value.name.value)
        if isinstance(value, ListValueNode):
            return GraphQLListValue(
                [self._convert_value(item) for item in value.values]
            )
        if isinstance(value, EnumValueNode):
            return GraphQLEnumValue(value.value)
        if isinstance(value, BooleanValueNode):
            return GraphQLBoolValue(value.value)
        if isinstance(value, ObjectValueNode):
            return GraphQLObjectValue(
                {
                    field.name.value: self._convert_value(field.value)
                    for field in value.fields
                }
            )
        raise ValueError(f"Unsupported type: {type(value)}")  # pragma: no cover

    def _convert_arguments(
        self, arguments: Iterable[ArgumentNode]
    ) -> list[GraphQLArgument]:
        return [
            GraphQLArgument(argument.name.value, self._convert_value(argument.value))
            for argument in arguments
        ]

    def _convert_directives(
        self, directives: Iterable[DirectiveNode]
    ) -> list[GraphQLDirective]:
        return [
            GraphQLDirective(
                directive.name.value,
                self._convert_arguments(directive.arguments),
            )
            for directive in directives
        ]

    def _convert_operation(
        self, operation_definition: OperationDefinitionNode
    ) -> GraphQLOperation:
        query_type = self._get_parent_type_by_name(
            operation_definition.operation.value.title()
        )

        assert operation_definition.name is not None
        operation_name = operation_definition.name.value
        result_class_name = f"{operation_name}Result"

        operation_type = self._collect_types(
            cast("HasSelectionSet", operation_definition),
            parent_type=query_type,
            class_name=result_class_name,
        )

        variables, variables_type = self._convert_variable_definitions(
            operation_definition.variable_definitions, operation_name=operation_name
        )

        return GraphQLOperation(
            operation_definition.name.value,
            kind=operation_definition.operation.value,
            selections=self._convert_selection_set(operation_definition.selection_set),
            directives=self._convert_directives(operation_definition.directives),
            variables=variables,
            type=cast("GraphQLObjectType", operation_type),
            variables_type=variables_type,
        )

    def _convert_variable_definitions(
        self,
        variable_definitions: Iterable[VariableDefinitionNode] | None,
        operation_name: str,
    ) -> tuple[list[GraphQLVariable], GraphQLObjectType | None]:
        if not variable_definitions:
            return [], None

        type_ = GraphQLObjectType(f"{operation_name}Variables", [])
        self._collect_type(type_)
        variables: list[GraphQLVariable] = []

        for variable_definition in variable_definitions:
            variable_type = self._collect_type_from_variable(variable_definition.type)
            variable = GraphQLVariable(
                variable_definition.variable.name.value,
                variable_type,
            )
            type_.fields.append(GraphQLField(variable.name, None, variable_type))
            variables.append(variable)

        return variables, type_

    def _get_operations(self, ast: DocumentNode) -> list[OperationDefinitionNode]:
        return [
            definition
            for definition in ast.definitions
            if isinstance(definition, OperationDefinitionNode)
        ]

    def _get_parent_type_by_name(self, name: str) -> GQLObjectType:
        parent_type = self.schema.type_map.get(name)
        assert isinstance(parent_type, GQLObjectType), (
            f"{name!r} is not a type in the graphql schema!"
        )
        return parent_type

    def _collect_type_from_variable(
        self, variable_type: TypeNode, parent_type: TypeNode | None = None
    ) -> GraphQLType:
        type_: GraphQLType | None = None

        if isinstance(variable_type, ListTypeNode):
            type_ = GraphQLList(
                self._collect_type_from_variable(variable_type.type, variable_type)
            )
        elif isinstance(variable_type, NonNullTypeNode):
            return self._collect_type_from_variable(variable_type.type, variable_type)
        elif isinstance(variable_type, NamedTypeNode):
            gql_type = self.schema.type_map.get(variable_type.name.value)
            assert gql_type is not None
            # Don't wrap in optional here - we handle wrapping below
            type_ = self._collect_type_from_gql_type(gql_type, wrap_optional=False)

        assert type_

        if parent_type is not None and isinstance(parent_type, NonNullTypeNode):
            return type_
        return GraphQLOptional(type_)

    def _collect_type_from_gql_type(
        self, gql_type: GraphQLOutputType, wrap_optional: bool = True
    ) -> GraphQLType:
        """Collect type from a graphql-core type.

        Args:
            gql_type: The graphql-core type to process
            wrap_optional: If True, wrap nullable types in GraphQLOptional.
                          This is needed for input object fields.
        """
        if isinstance(gql_type, GraphQLNonNull):
            # Non-null types are not optional
            return self._collect_type_from_gql_type(gql_type.of_type, wrap_optional=False)
        if isinstance(gql_type, GQLList):
            inner = self._collect_type_from_gql_type(gql_type.of_type, wrap_optional=True)
            result = GraphQLList(inner)
            return GraphQLOptional(result) if wrap_optional else result
        if isinstance(gql_type, GraphQLScalarType):
            result = self._collect_scalar(gql_type.name)
            return GraphQLOptional(result) if wrap_optional else result
        if isinstance(gql_type, GraphQLEnumType):
            result = self._collect_enum(gql_type)
            return GraphQLOptional(result) if wrap_optional else result
        if isinstance(gql_type, (GQLObjectType, GraphQLInputObjectType)):
            type_ = GraphQLObjectType(gql_type.name, [])
            for graphql_field_name, field in gql_type.fields.items():
                # Input object fields need to wrap optional
                field_type = self._collect_type_from_gql_type(field.type, wrap_optional=True)
                python_field_name = self._get_field_name(field, graphql_field_name)
                # Get default value if present
                default_value = None
                # Try Strawberry definition first (preserves UNSET)
                ext = getattr(field, "extensions", {})
                strawberry_def = ext.get("strawberry-definition")
                if strawberry_def is not None and hasattr(strawberry_def, "default"):
                    if strawberry_def.default is not MISSING:
                        default_value = _py_to_graphql_value(strawberry_def.default)
                else:
                    # Fall back to graphql-core default_value for SDL schemas
                    field_default = getattr(field, "default_value", Undefined)
                    if field_default is not Undefined:
                        default_value = _py_to_graphql_value(field_default)
                type_.fields.append(GraphQLField(python_field_name, None, field_type, default_value=default_value))
            self._collect_type(type_)
            return GraphQLOptional(type_) if wrap_optional else type_
        if isinstance(gql_type, GraphQLUnionType):
            union_types = [
                cast(
                    "GraphQLObjectType",
                    self._collect_type_from_gql_type(member_type, wrap_optional=False),
                )
                for member_type in gql_type.types
            ]
            union = GraphQLUnion(gql_type.name, union_types)
            self._collect_type(union)
            return GraphQLOptional(union) if wrap_optional else union
        raise ValueError(f"Unsupported type: {gql_type}")  # pragma: no cover

    def _get_field_type_from_gql(self, gql_type: GraphQLOutputType) -> GraphQLType:
        if isinstance(gql_type, GraphQLNonNull):
            return self._get_inner_field_type(gql_type.of_type)
        return GraphQLOptional(self._get_inner_field_type(gql_type))

    def _get_inner_field_type(self, gql_type: GraphQLOutputType) -> GraphQLType:
        if isinstance(gql_type, GraphQLNonNull):
            return self._get_inner_field_type(gql_type.of_type)
        if isinstance(gql_type, GQLList):
            inner = self._get_field_type_from_gql(gql_type.of_type)
            return GraphQLList(inner)
        if isinstance(gql_type, GraphQLScalarType):
            return self._collect_scalar(gql_type.name)
        if isinstance(gql_type, GraphQLEnumType):
            return self._collect_enum(gql_type)
        raise ValueError(f"Unsupported type: {gql_type}")  # pragma: no cover

    def _field_from_selection(
        self, selection: FieldNode, parent_type: GQLObjectType
    ) -> GraphQLField:
        if selection.name.value == "__typename":
            return GraphQLField("__typename", None, GraphQLScalar("String", None))
        field = parent_type.fields.get(selection.name.value)
        assert field, f"{parent_type.name},{selection.name.value}"
        field_type = self._get_field_type_from_gql(field.type)
        field_name = self._get_field_name(field, selection.name.value)
        return GraphQLField(
            field_name,
            selection.alias.value if selection.alias else None,
            field_type,
        )

    def _field_from_selection_set(
        self,
        selection: FieldNode,
        class_name: str,
        parent_type: GQLObjectType,
    ) -> GraphQLField:
        assert selection.selection_set is not None

        selected_field = parent_type.fields.get(selection.name.value)
        assert selected_field, f"Couldn't find {parent_type.name}.{selection.name.value}"

        base_type = selected_field.type
        is_nullable = True
        wrappers: list[type] = []

        while True:
            if isinstance(base_type, GraphQLNonNull):
                is_nullable = False
                base_type = base_type.of_type
            elif isinstance(base_type, GQLList):
                # Add List wrapper first
                wrappers.append(GraphQLList)
                # If the current level is nullable, wrap in Optional AFTER List
                if is_nullable:
                    wrappers.append(GraphQLOptional)
                is_nullable = True  # Reset for inner type
                base_type = base_type.of_type
            else:
                break

        wrapper: Callable[[GraphQLType], GraphQLType] | None = None
        if wrappers:
            # Apply wrappers in order (NOT reversed) - inner wrappers first
            # For [Person!] -> wrappers = [GraphQLList, GraphQLOptional]
            # Apply: Person -> list[Person] -> Optional[list[Person]]
            def make_wrapper(
                wrappers: list[type],
            ) -> Callable[[GraphQLType], GraphQLType]:
                def apply(t: GraphQLType) -> GraphQLType:
                    for w in wrappers:
                        t = w(t)
                    return t

                return apply

            wrapper = make_wrapper(wrappers)

        name = capitalize_first(to_camel_case(selection.name.value))
        class_name = f"{class_name}{name}"

        field_type: GraphQLType

        if isinstance(base_type, (GraphQLUnionType, GraphQLInterfaceType)):
            field_type = self._collect_types_with_inline_fragments(
                selection, parent_type, class_name
            )
        else:
            assert isinstance(base_type, GQLObjectType), (
                f"Expected object type, got {base_type}"
            )
            field_type = self._collect_types(selection, base_type, class_name)

        if is_nullable:
            field_type = GraphQLOptional(field_type)
        if wrapper:
            field_type = wrapper(field_type)

        field_name = self._get_field_name(selected_field, selection.name.value)
        return GraphQLField(
            field_name,
            selection.alias.value if selection.alias else None,
            field_type,
        )

    def _get_field(
        self,
        selection: FieldNode,
        class_name: str,
        parent_type: GQLObjectType,
    ) -> GraphQLField:
        if selection.selection_set:
            return self._field_from_selection_set(selection, class_name, parent_type)
        return self._field_from_selection(selection, parent_type)

    def _collect_types_with_inline_fragments(
        self,
        selection: HasSelectionSet,
        parent_type: GQLObjectType,
        class_name: str,
    ) -> GraphQLObjectType | GraphQLUnion:
        sub_types = self._collect_types_using_fragments(
            selection, parent_type, class_name
        )
        if len(sub_types) == 1:
            return sub_types[0]
        union = GraphQLUnion(class_name, sub_types)
        self._collect_type(union)
        return union

    def _collect_types(
        self,
        selection: HasSelectionSet,
        parent_type: GQLObjectType,
        class_name: str,
        graph_ql_object_type_factory: Callable[
            [str], GraphQLObjectType
        ] = GraphQLObjectType,
    ) -> GraphQLType:
        assert selection.selection_set is not None
        selection_set = selection.selection_set

        if any(
            isinstance(selection, InlineFragmentNode)
            for selection in selection_set.selections
        ):
            return self._collect_types_with_inline_fragments(
                selection, parent_type, class_name
            )

        current_type = graph_ql_object_type_factory(class_name)
        fields: list[GraphQLFragmentSpread | GraphQLField] = []

        for sub_selection in selection_set.selections:
            if isinstance(sub_selection, FragmentSpreadNode):
                fields.append(GraphQLFragmentSpread(sub_selection.name.value))
                continue
            assert isinstance(sub_selection, FieldNode)
            field = self._get_field(sub_selection, class_name, parent_type)
            fields.append(field)

        if any(isinstance(f, GraphQLFragmentSpread) for f in fields):
            if len(fields) > 1:
                raise ValueError(
                    "Queries with Fragments cannot currently include separate fields."
                )
            spread_field = fields[0]
            assert isinstance(spread_field, GraphQLFragmentSpread)
            return next(
                t
                for t in self.types
                if isinstance(t, GraphQLObjectType) and t.name == spread_field.name
            )

        current_type.fields = cast("list[GraphQLField]", fields)
        self._collect_type(current_type)
        return current_type

    def generate_code(self) -> CodegenResult:
        return self.plugin_manager.generate_code(
            types=self.types, operation=self.operation
        )

    def _collect_types_using_fragments(
        self,
        selection: HasSelectionSet,
        parent_type: GQLObjectType,
        class_name: str,
    ) -> list[GraphQLObjectType]:
        assert selection.selection_set

        common_fields: list[GraphQLField] = []
        fragments: list[InlineFragmentNode] = []
        sub_types: list[GraphQLObjectType] = []

        for sub_selection in selection.selection_set.selections:
            if isinstance(sub_selection, FieldNode):
                common_fields.append(
                    self._get_field(sub_selection, class_name, parent_type)
                )
            if isinstance(sub_selection, InlineFragmentNode):
                fragments.append(sub_selection)

        all_common_fields_typename = all(f.name == "__typename" for f in common_fields)

        for fragment in fragments:
            type_condition_name = fragment.type_condition.name.value
            fragment_class_name = class_name + type_condition_name

            current_type = GraphQLObjectType(
                fragment_class_name,
                list(common_fields),
                graphql_typename=type_condition_name,
            )
            fields: list[GraphQLFragmentSpread | GraphQLField] = []

            for sub_selection in fragment.selection_set.selections:
                if isinstance(sub_selection, FragmentSpreadNode):
                    fields.append(GraphQLFragmentSpread(sub_selection.name.value))
                    continue

                assert isinstance(sub_selection, FieldNode)

                fragment_parent_type = self._get_parent_type_by_name(type_condition_name)

                fields.append(
                    self._get_field(
                        selection=sub_selection,
                        class_name=fragment_class_name,
                        parent_type=fragment_parent_type,
                    )
                )

            if any(isinstance(f, GraphQLFragmentSpread) for f in fields):
                if len(fields) > 1:
                    raise ValueError(
                        "Queries with Fragments cannot include separate fields."
                    )
                spread_field = fields[0]
                assert isinstance(spread_field, GraphQLFragmentSpread)
                sub_type = next(
                    t
                    for t in self.types
                    if isinstance(t, GraphQLObjectType) and t.name == spread_field.name
                )
                fields = [*sub_type.fields]
                if all_common_fields_typename:
                    sub_types.append(sub_type)
                    continue

            current_type.fields.extend(cast("list[GraphQLField]", fields))
            sub_types.append(current_type)

        # If there are no inline fragments, just return a single type with common fields
        if not sub_types and common_fields:
            single_type = GraphQLObjectType(class_name, common_fields)
            self._collect_type(single_type)
            return [single_type]

        for sub_type in sub_types:
            self._collect_type(sub_type)

        return sub_types

    def _collect_scalar(self, name: str) -> GraphQLScalar:
        # Look up Python type from registry if available
        python_type = self.scalar_registry.get(name)
        graphql_scalar = GraphQLScalar(name, python_type=python_type)
        self._collect_type(graphql_scalar)
        return graphql_scalar

    def _collect_enum(self, enum_type: GraphQLEnumType) -> GraphQLEnum:
        enum_values = list(enum_type.values.keys())
        # Look up Python type from registry if available
        python_type = self.enum_registry.get(enum_type.name)
        if python_type is None:
            # Create a dynamic enum for SDL-based codegen
            python_type = type(Enum(enum_type.name, {v: v for v in enum_values}))
        graphql_enum = GraphQLEnum(
            enum_type.name,
            enum_values,
            python_type=python_type,
        )
        self._collect_type(graphql_enum)
        return graphql_enum


# Backwards compatibility alias
SDLQueryCodegen = QueryCodegen


__all__ = [
    "CodegenFile",
    "CodegenResult",
    "ConsolePlugin",
    "QueryCodegen",
    "QueryCodegenPlugin",
    "SDLQueryCodegen",
]
