import subprocess
import os
import tempfile

def compile_latex_to_pdf(latex_code: str, output_dir: str = None) -> str:
    """
    Compiles the provided LaTeX code into a PDF file using pdflatex.

    Args:
        latex_code: A string containing the complete LaTeX code.
        output_dir: Directory to write the output PDF; if not provided, a temporary directory is used.

    Returns:
        The path to the generated PDF file.
    """
    if output_dir is None:
        output_dir = tempfile.mkdtemp()

    tex_path = os.path.join(output_dir, "document.tex")
    pdf_path = os.path.join(output_dir, "document.pdf")

    # Write the LaTeX code to a .tex file.
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(latex_code)

    # Run pdflatex; run twice to ensure all references are resolved.
    try:
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-output-directory", output_dir, tex_path],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-output-directory", output_dir, tex_path],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"LaTeX compilation failed: {e}")

    if not os.path.exists(pdf_path):
        raise FileNotFoundError("PDF file was not generated.")

    return pdf_path
