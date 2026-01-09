from pathlib import Path

import pytest
from typer import Typer
from typer.testing import CliRunner

from strawberry.cli.commands.codegen import ConsolePlugin
from strawberry.codegen import CodegenFile, CodegenResult, QueryCodegenPlugin
from strawberry.codegen.types import GraphQLOperation, GraphQLType


class ConsoleTestPlugin(ConsolePlugin):
    def on_end(self, result: CodegenResult):
        result.files[0].path = "renamed.py"

        return super().on_end(result)


class QueryCodegenTestPlugin(QueryCodegenPlugin):
    def generate_code(
        self, types: list[GraphQLType], operation: GraphQLOperation
    ) -> list[CodegenFile]:
        return [
            CodegenFile(
                path="test.py",
                content=f"# This is a test file for {operation.name}",
            )
        ]


class EmptyPlugin(QueryCodegenPlugin):
    def generate_code(
        self, types: list[GraphQLType], operation: GraphQLOperation
    ) -> list[CodegenFile]:
        return [
            CodegenFile(
                path="test.py",
                content="# Empty",
            )
        ]


@pytest.fixture
def query_file_path(tmp_path: Path) -> Path:
    output_path = tmp_path / "query.graphql"
    output_path.write_text(
        """
        query GetUser {
            user {
                name
            }
        }
        """
    )
    return output_path


@pytest.fixture
def query_file_path2(tmp_path: Path) -> Path:
    output_path = tmp_path / "query2.graphql"
    output_path.write_text(
        """
        query GetUser {
            user {
                name
            }
        }
        """
    )
    return output_path


def test_codegen(
    cli_app: Typer, cli_runner: CliRunner, query_file_path: Path, tmp_path: Path
):
    selector = "tests.fixtures.sample_package.sample_module:schema"
    result = cli_runner.invoke(
        cli_app,
        [
            "codegen",
            "-p",
            "tests.cli.test_codegen:QueryCodegenTestPlugin",
            "-o",
            str(tmp_path),
            "--schema",
            selector,
            str(query_file_path),
        ],
    )

    assert result.exit_code == 0

    code_path = tmp_path / "test.py"

    assert code_path.exists()
    assert code_path.read_text() == "# This is a test file for GetUser"


def test_codegen_multiple_files(
    cli_app: Typer,
    cli_runner: CliRunner,
    query_file_path: Path,
    query_file_path2: Path,
    tmp_path: Path,
):
    expected_paths = [
        tmp_path / "query.py",
        tmp_path / "query2.py",
        tmp_path / "query.ts",
        tmp_path / "query2.ts",
    ]
    for path in expected_paths:
        assert not path.exists()

    selector = "tests.fixtures.sample_package.sample_module:schema"
    result = cli_runner.invoke(
        cli_app,
        [
            "codegen",
            "-p",
            "python",
            "-p",
            "typescript",
            "-o",
            str(tmp_path),
            "--schema",
            selector,
            str(query_file_path),
            str(query_file_path2),
        ],
    )

    assert result.exit_code == 0

    for path in expected_paths:
        assert path.exists()
        assert " GetUserResult" in path.read_text()


def test_codegen_pass_no_query(cli_app: Typer, cli_runner: CliRunner, tmp_path: Path):
    selector = "tests.fixtures.sample_package.sample_module:schema"
    result = cli_runner.invoke(
        cli_app,
        [
            "codegen",
            "-p",
            "tests.cli.test_codegen:EmptyPlugin",
            "-o",
            str(tmp_path),
            "--schema",
            selector,
        ],
    )

    assert result.exit_code == 0


def test_codegen_passing_plugin_symbol(
    cli_app: Typer, cli_runner: CliRunner, query_file_path: Path, tmp_path: Path
):
    selector = "tests.fixtures.sample_package.sample_module:schema"
    result = cli_runner.invoke(
        cli_app,
        [
            "codegen",
            "-p",
            "tests.cli.test_codegen:EmptyPlugin",
            "-o",
            str(tmp_path),
            "--schema",
            selector,
            str(query_file_path),
        ],
    )

    assert result.exit_code == 0

    code_path = tmp_path / "test.py"

    assert code_path.exists()
    assert code_path.read_text() == "# Empty"


def test_codegen_returns_error_when_symbol_does_not_exist(
    cli_app: Typer, cli_runner: CliRunner, query_file_path: Path, tmp_path: Path
):
    selector = "tests.fixtures.sample_package.sample_module:schema"
    result = cli_runner.invoke(
        cli_app,
        [
            "codegen",
            "-p",
            "tests.cli.test_codegen:SomePlugin",
            "--schema",
            selector,
            "-o",
            str(tmp_path),
            str(query_file_path),
        ],
    )

    assert result.exit_code == 1
    assert result.exception
    assert result.exception.args == (
        "module 'tests.cli.test_codegen' has no attribute 'SomePlugin'",
    )


def test_codegen_returns_error_when_module_does_not_exist(
    cli_app: Typer, cli_runner: CliRunner, query_file_path: Path, tmp_path: Path
):
    selector = "tests.fixtures.sample_package.sample_module:schema"
    result = cli_runner.invoke(
        cli_app,
        [
            "codegen",
            "-p",
            "fake_module_plugin",
            "--schema",
            selector,
            "-o",
            str(tmp_path),
            str(query_file_path),
        ],
    )

    assert result.exit_code == 1
    assert "Error: Plugin fake_module_plugin not found" in result.output


def test_codegen_with_sdl_flag(
    cli_app: Typer, cli_runner: CliRunner, query_file_path: Path, tmp_path: Path
):
    # Create an SDL file with a simple schema
    sdl_file_path = tmp_path / "schema.graphql"
    sdl_file_path.write_text(
        """
        type Query {
            user: User
        }

        type User {
            id: ID!
            name: String!
            email: String
        }
        """
    )

    result = cli_runner.invoke(
        cli_app,
        [
            "codegen",
            "-p",
            "tests.cli.test_codegen:QueryCodegenTestPlugin",
            "-o",
            str(tmp_path),
            "--sdl",
            str(sdl_file_path),
            str(query_file_path),
        ],
    )

    assert result.exit_code == 0

    code_path = tmp_path / "test.py"
    assert code_path.exists()
    assert code_path.read_text() == "# This is a test file for GetUser"


def test_codegen_schema_and_sdl_mutually_exclusive(
    cli_app: Typer, cli_runner: CliRunner, query_file_path: Path, tmp_path: Path
):
    # Create an SDL file
    sdl_file_path = tmp_path / "schema.graphql"
    sdl_file_path.write_text("type Query { hello: String }")

    selector = "tests.fixtures.sample_package.sample_module:schema"

    # Try to use both --schema and --sdl
    result = cli_runner.invoke(
        cli_app,
        [
            "codegen",
            "-p",
            "tests.cli.test_codegen:QueryCodegenTestPlugin",
            "-o",
            str(tmp_path),
            "--schema",
            selector,
            "--sdl",
            str(sdl_file_path),
            str(query_file_path),
        ],
    )

    assert result.exit_code == 1
    assert "--schema and --sdl are mutually exclusive" in result.output


def test_codegen_requires_schema_or_sdl(
    cli_app: Typer, cli_runner: CliRunner, query_file_path: Path, tmp_path: Path
):
    # Try to run without --schema or --sdl
    result = cli_runner.invoke(
        cli_app,
        [
            "codegen",
            "-p",
            "tests.cli.test_codegen:QueryCodegenTestPlugin",
            "-o",
            str(tmp_path),
            str(query_file_path),
        ],
    )

    assert result.exit_code == 1
    assert "Either --schema or --sdl must be provided" in result.output


def test_codegen_returns_error_when_does_not_find_plugin(
    cli_app: Typer, cli_runner: CliRunner, query_file_path: Path, tmp_path: Path
):
    selector = "tests.fixtures.sample_package.sample_module:schema"
    result = cli_runner.invoke(
        cli_app,
        [
            "codegen",
            "-p",
            "tests.cli.test_server",
            "--schema",
            selector,
            "-o",
            str(tmp_path),
            str(query_file_path),
        ],
    )

    assert result.exit_code == 1
    assert "Error: Plugin tests.cli.test_server not found" in result.output


def test_codegen_finds_our_plugins(
    cli_app: Typer, cli_runner: CliRunner, query_file_path: Path, tmp_path: Path
):
    selector = "tests.fixtures.sample_package.sample_module:schema"
    result = cli_runner.invoke(
        cli_app,
        [
            "codegen",
            "-p",
            "python",
            "--schema",
            selector,
            "-o",
            str(tmp_path),
            str(query_file_path),
        ],
    )

    assert result.exit_code == 0

    code_path = tmp_path / query_file_path.with_suffix(".py").name

    assert code_path.exists()
    assert "class GetUserResult" in code_path.read_text()


def test_can_use_custom_cli_plugin(
    cli_app: Typer, cli_runner: CliRunner, query_file_path: Path, tmp_path: Path
):
    selector = "tests.fixtures.sample_package.sample_module:schema"
    result = cli_runner.invoke(
        cli_app,
        [
            "codegen",
            "--cli-plugin",
            "tests.cli.test_codegen:ConsoleTestPlugin",
            "-p",
            "python",
            "--schema",
            selector,
            "-o",
            str(tmp_path),
            str(query_file_path),
        ],
    )

    assert result.exit_code == 0

    code_path = tmp_path / "renamed.py"

    assert code_path.exists()
    assert "class GetUserResult" in code_path.read_text()


def test_codegen_with_sdl_flag_using_builtin_plugins(
    cli_app: Typer, cli_runner: CliRunner, tmp_path: Path
):
    """Test that --sdl works with built-in plugins like python and typescript."""
    # Create an SDL file with a schema that matches the query
    sdl_file_path = tmp_path / "schema.graphql"
    sdl_file_path.write_text(
        """
        type Query {
            user: User
        }

        type User {
            id: ID!
            name: String!
            email: String
        }
        """
    )

    # Create a query file
    query_file_path = tmp_path / "query.graphql"
    query_file_path.write_text(
        """
        query GetUser {
            user {
                id
                name
                email
            }
        }
        """
    )

    result = cli_runner.invoke(
        cli_app,
        [
            "codegen",
            "-p",
            "python",
            "-o",
            str(tmp_path),
            "--sdl",
            str(sdl_file_path),
            str(query_file_path),
        ],
    )

    assert result.exit_code == 0

    code_path = tmp_path / "query.py"
    assert code_path.exists()
    content = code_path.read_text()
    assert "class GetUserResult" in content
    assert "GetUserResultUser" in content


def test_codegen_with_sdl_flag_invalid_sdl(
    cli_app: Typer, cli_runner: CliRunner, tmp_path: Path
):
    """Test that --sdl provides meaningful error for invalid SDL."""
    # Create an SDL file with invalid syntax
    sdl_file_path = tmp_path / "schema.graphql"
    sdl_file_path.write_text("this is not valid SDL {{{")

    query_file_path = tmp_path / "query.graphql"
    query_file_path.write_text("query Test { hello }")

    result = cli_runner.invoke(
        cli_app,
        [
            "codegen",
            "-p",
            "tests.cli.test_codegen:QueryCodegenTestPlugin",
            "-o",
            str(tmp_path),
            "--sdl",
            str(sdl_file_path),
            str(query_file_path),
        ],
    )

    # Should fail due to invalid SDL
    assert result.exit_code == 1


def test_codegen_with_sdl_nested_types(
    cli_app: Typer, cli_runner: CliRunner, tmp_path: Path
):
    """Test SDL codegen with nested object types."""
    sdl_file_path = tmp_path / "schema.graphql"
    sdl_file_path.write_text(
        """
        type Query {
            user: User
        }

        type User {
            id: ID!
            name: String!
            posts: [Post!]!
        }

        type Post {
            id: ID!
            title: String!
            author: User!
        }
        """
    )

    query_file_path = tmp_path / "query.graphql"
    query_file_path.write_text(
        """
        query GetUserWithPosts {
            user {
                id
                name
                posts {
                    id
                    title
                }
            }
        }
        """
    )

    result = cli_runner.invoke(
        cli_app,
        [
            "codegen",
            "-p",
            "python",
            "-o",
            str(tmp_path),
            "--sdl",
            str(sdl_file_path),
            str(query_file_path),
        ],
    )

    assert result.exit_code == 0

    code_path = tmp_path / "query.py"
    assert code_path.exists()
    content = code_path.read_text()
    assert "GetUserWithPostsResult" in content
    assert "GetUserWithPostsResultUser" in content
    assert "GetUserWithPostsResultUserPosts" in content


def test_codegen_with_sdl_enum_types(
    cli_app: Typer, cli_runner: CliRunner, tmp_path: Path
):
    """Test SDL codegen with enum types."""
    sdl_file_path = tmp_path / "schema.graphql"
    sdl_file_path.write_text(
        """
        type Query {
            user: User
        }

        type User {
            id: ID!
            name: String!
            status: Status!
        }

        enum Status {
            ACTIVE
            INACTIVE
            PENDING
        }
        """
    )

    query_file_path = tmp_path / "query.graphql"
    query_file_path.write_text(
        """
        query GetUserStatus {
            user {
                id
                status
            }
        }
        """
    )

    result = cli_runner.invoke(
        cli_app,
        [
            "codegen",
            "-p",
            "python",
            "-o",
            str(tmp_path),
            "--sdl",
            str(sdl_file_path),
            str(query_file_path),
        ],
    )

    assert result.exit_code == 0

    code_path = tmp_path / "query.py"
    assert code_path.exists()
    content = code_path.read_text()
    assert "Status" in content
    assert "ACTIVE" in content
    assert "INACTIVE" in content
    assert "PENDING" in content


def test_codegen_with_sdl_nullable_fields(
    cli_app: Typer, cli_runner: CliRunner, tmp_path: Path
):
    """Test SDL codegen handles nullable and non-nullable fields correctly."""
    sdl_file_path = tmp_path / "schema.graphql"
    sdl_file_path.write_text(
        """
        type Query {
            user: User
        }

        type User {
            id: ID!
            name: String!
            nickname: String
            age: Int
        }
        """
    )

    query_file_path = tmp_path / "query.graphql"
    query_file_path.write_text(
        """
        query GetUser {
            user {
                id
                name
                nickname
                age
            }
        }
        """
    )

    result = cli_runner.invoke(
        cli_app,
        [
            "codegen",
            "-p",
            "python",
            "-o",
            str(tmp_path),
            "--sdl",
            str(sdl_file_path),
            str(query_file_path),
        ],
    )

    assert result.exit_code == 0

    code_path = tmp_path / "query.py"
    assert code_path.exists()
    content = code_path.read_text()
    # Non-nullable fields should not have Optional
    assert "id: str" in content
    assert "name: str" in content
    # Nullable fields should have Optional
    assert "Optional[str]" in content
    assert "Optional[int]" in content


def test_codegen_with_sdl_query_variables(
    cli_app: Typer, cli_runner: CliRunner, tmp_path: Path
):
    """Test SDL codegen with query variables."""
    sdl_file_path = tmp_path / "schema.graphql"
    sdl_file_path.write_text(
        """
        type Query {
            user(id: ID!): User
        }

        type User {
            id: ID!
            name: String!
        }
        """
    )

    query_file_path = tmp_path / "query.graphql"
    query_file_path.write_text(
        """
        query GetUserById($userId: ID!) {
            user(id: $userId) {
                id
                name
            }
        }
        """
    )

    result = cli_runner.invoke(
        cli_app,
        [
            "codegen",
            "-p",
            "python",
            "-o",
            str(tmp_path),
            "--sdl",
            str(sdl_file_path),
            str(query_file_path),
        ],
    )

    assert result.exit_code == 0

    code_path = tmp_path / "query.py"
    assert code_path.exists()
    content = code_path.read_text()
    assert "GetUserByIdVariables" in content
    assert "userId" in content
