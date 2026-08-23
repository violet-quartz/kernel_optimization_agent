from anthropic import Anthropic, beta_tool
import subprocess
import os
import sys
import tempfile
from dotenv import load_dotenv


user_input = """
读取文件 /Users/flora/code/kernelswift_competition/v0/centre_random_augmentation.py， 介绍一下它在干什么。
以 get_init_inputs 的返回值初始化 Model，以 get_inputs 的返回值作为 forwad 的 参数，运行 model.forward，看输出是什么。
"""

load_dotenv()
client = Anthropic()

@beta_tool
def read_file(file_path: str) -> str:
    """Read a file, get its content.

    Args:
        file_path: file path
    Returns:
        A string with all the file content
    """
    with open(file_path, mode="r", encoding="utf-8") as file:
        return file.read()

@beta_tool
def run_python(code: str) -> str:
    """Execute a snippet of Python code in a subprocess and return its output.

    Args:
        code: The Python source code to execute.
    Returns:
        Combined stdout and stderr, with the exit code appended on failure.
    """
    with tempfile.NamedTemporaryFile(
        "w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(code)
        path = f.name
    try:
        result = subprocess.run(
            [sys.executable, path],
            capture_output=True,
            text=True,
            timeout=300
        )
    except subprocess.TimeoutExpired as e:
        return f"[timeout after 300s]\n{e.stdout or ''}{e.stderr or ''}"
    finally:
        os.unlink(path)
    out = result.stdout + result.stderr
    if result.returncode != 0:
        out += f"\n[exit code {result.returncode}]"
    return out[-20000:] if len(out) > 20000 else out or "(no output)"

messages = [{"role": "user", "content": user_input}]
Tools = [read_file, run_python]

runner = client.beta.messages.tool_runner(
    max_tokens=1024,
    model="deepseek-v4-pro",
    tools=Tools,
    messages=[
        {"role": "user", "content": user_input},
    ],
)
for message in runner:
    print(message)
