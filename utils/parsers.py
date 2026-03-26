#!/usr/bin/env python3
"""
Parsers for extracting information from instruction files.
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple


# Mapping from code fence language to file extension
LANGUAGE_TO_EXTENSION = {
    'python': '.py',
    'html': '.html',
    'text': '.txt',
    'markdown': '.md',
    'sql': '.sql',
    'yaml': '.yaml',
    'yml': '.yml',
    'json': '.json',
    'css': '.css',
    'mermaid': '.mmd',
    'bash': '.sh',
    'sh': '.sh',
    'dockerfile': '',  # Dockerfile has no extension
    'protobuf': '.pbtxt',
    'unknown_language': '.txt',  # fallback
}


def extract_dependencies(instruction_content: str) -> List[str]:
    """
    Parse the <dependencies> section and extract pip package specifications.

    Args:
        instruction_content: Full content of the instruction file

    Returns:
        List of pip package specifications (e.g., ["pydantic~=2.10.6", "openai~=1.66.3"])
    """
    # Find the dependencies section
    deps_match = re.search(r'<dependencies>\s*<dependencies>(.*?)</dependencies>\s*</dependencies>',
                          instruction_content, re.DOTALL)

    if not deps_match:
        # Try alternative pattern (single dependencies tag)
        deps_match = re.search(r'<dependencies>(.*?)</dependencies>',
                              instruction_content, re.DOTALL)

    if not deps_match:
        print("Warning: No <dependencies> section found")
        return []

    deps_section = deps_match.group(1)

    # Parse package names and versions
    # Pattern: ## package_name followed by Version: version_spec
    packages = []

    # Split by ## to get each package section
    package_sections = re.split(r'\n##\s+', deps_section)

    for section in package_sections:
        if not section.strip():
            continue

        # Extract package name (first line or first word)
        lines = section.strip().split('\n')
        if not lines:
            continue

        package_name = lines[0].strip()

        # Skip if it looks like markdown headers or other content
        if not package_name or package_name.startswith('#'):
            continue

        # Extract version if present
        version_match = re.search(r'Version:\s*([~=<>!]+[\d.]+[\w.-]*)', section, re.IGNORECASE)

        if version_match:
            version_spec = version_match.group(1)
            packages.append(f"{package_name}{version_spec}")
        else:
            # No version specified, just add package name
            packages.append(package_name)

    print(f"Extracted {len(packages)} dependencies")
    return packages


def extract_implementations(instruction_content: str) -> Dict[str, str]:
    """
    Parse the <implementations> section and extract file contents.
    Handles all file types (Python, HTML, JSON, YAML, etc.)

    Args:
        instruction_content: Full content of the instruction file

    Returns:
        Dictionary mapping file paths to their contents
    """
    # Find the implementations section
    impl_match = re.search(r'<implementations>(.*?)</implementations>',
                          instruction_content, re.DOTALL)

    if not impl_match:
        raise ValueError("Could not find <implementations> section")

    impl_section = impl_match.group(1)

    # Parse file sections
    # Pattern: ## file/path followed by ```language code ```
    files = {}

    # Find all file sections - capture language marker if present
    # Updated pattern to match ``` only at start of line (with optional whitespace)
    # This prevents matching ``` that appears in strings within the code
    # Updated to handle empty files (where content might be empty between backticks)
    file_pattern = r'##\s+([^\n]+)\s*\n\s*```([a-z_]*)\s*\n(.*?)\n?\s*```'

    matches = re.finditer(file_pattern, impl_section, re.DOTALL)

    for match in matches:
        file_path = match.group(1).strip()
        language = match.group(2).strip() or 'python'  # default to python if no language
        file_content = match.group(3)

        # Clean up file path (remove any markdown formatting)
        file_path = file_path.strip('`').strip()

        # If file path doesn't have an extension, try to add one based on language
        if '.' not in file_path.split('/')[-1]:  # Check if filename has no extension
            # Special cases
            if file_path.endswith('Dockerfile'):
                # Dockerfile doesn't get an extension
                pass
            elif language in LANGUAGE_TO_EXTENSION:
                extension = LANGUAGE_TO_EXTENSION[language]
                if extension:  # Only add if extension is not empty string
                    file_path += extension

        files[file_path] = file_content

    print(f"Extracted {len(files)} file implementations")

    # Debug: Print file types found
    file_types = {}
    for path in files.keys():
        ext = Path(path).suffix or 'no-ext'
        file_types[ext] = file_types.get(ext, 0) + 1
    print(f"  File types: {dict(file_types)}")

    return files


def modify_file_content(code: str, file_path: str) -> str:
    """
    Modify file content based on file type:
    - Python files: Remove imports and replace function bodies with NotImplementedError
    - Non-Python files: Return as-is

    Args:
        code: Original code
        file_path: File path (to determine type)

    Returns:
        Modified code
    """
    from pathlib import Path

    # Only modify Python files
    if not file_path.endswith('.py'):
        return code

    import ast

    try:
        tree = ast.parse(code)
    except SyntaxError:
        # If parsing fails, return original code
        print(f"Warning: Could not parse Python code in {file_path}, returning original")
        return code

    class CodeModifier(ast.NodeTransformer):
        def visit_Import(self, node):
            # Remove import statements
            return None

        def visit_ImportFrom(self, node):
            # Remove from...import statements
            return None

        def visit_FunctionDef(self, node):
            # Replace function bodies with NotImplementedError
            new_body = []

            # Keep docstring if it exists
            if (node.body and
                isinstance(node.body[0], ast.Expr) and
                isinstance(node.body[0].value, ast.Constant) and
                isinstance(node.body[0].value.value, str)):
                new_body.append(node.body[0])

            # Add raise NotImplementedError
            new_body.append(
                ast.Raise(
                    exc=ast.Call(
                        func=ast.Name(id='NotImplementedError', ctx=ast.Load()),
                        args=[],
                        keywords=[]
                    ),
                    cause=None
                )
            )

            node.body = new_body
            # Continue visiting nested functions/classes
            self.generic_visit(node)
            return node

        def visit_AsyncFunctionDef(self, node):
            return self.visit_FunctionDef(node)

    modifier = CodeModifier()
    new_tree = modifier.visit(tree)
    ast.fix_missing_locations(new_tree)

    return ast.unparse(new_tree)


def extract_readme(instruction_content: str) -> str:
    """
    Extract the README section from instruction content.

    Args:
        instruction_content: Full content of the instruction file

    Returns:
        README content
    """
    readme_match = re.search(r'<readme>(.*?)</readme>',
                            instruction_content, re.DOTALL)

    if readme_match:
        return readme_match.group(1).strip()

    return ""
