import requests
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import OllamaLLM


def escape_for_ollama(text: str) -> str:
    """
    Escape a multiline string so that Ollama doesn't interpret any curly brace fields.
    This is done by doubling all curly braces.
    """
    return text.replace("{", "{{").replace("}", "}}")

def get_ai_summary(
    job_description: str,
    base_url: str = "http://localhost:11434",
    model: str = "llama3.1:8b-64k",
    timeout: int = 600,
) -> str:
    """
    Send text to an AI model using LangChain and get a summary.

    Args:
        text: The text to summarize
        base_url: Base URL for the API (Ollama or OpenAI)
        model: The model to use for summarization
        timeout: Timeout in seconds for the request
        api_key: API key for authentication (required for OpenAI)

    Returns:
        A summary of the text
    """
    try:
        print(f"Setting up Ollama model '{model}' at {base_url}")
        # pylint: disable=not-callable
        llm = OllamaLLM(
            base_url=base_url,
            model=model,
            timeout=timeout,
            model_kwargs={
                "temperature": 0.2,
                "num_gpu": 999,  # Request maximum GPU layers
                "num_cpu": 0,  # Don't use CPU cores
                "mirostat": 0,  # Disable adaptive sampling which can use CPU
            },
        )

        # Create a prompt template for summarization
        prompt = PromptTemplate.from_template(escape_for_ollama(
            r"""Given the following resume in latex, craft a STRONG one sentence summary adhering to the job description below that I can display in my resume. Also list a new technical skills section that adheres to the position overview below, including the name of the company and position if possible. The one sentence summary should be professional and advertise myself towards the position at hand.

resume:            
\documentclass[a4paper,10pt]{article}
\usepackage[left=0.6in, right=0.6in, top=0.6in, bottom=0.6in]{geometry}
\usepackage{enumitem, hyperref, fontawesome, titlesec}
\usepackage{xcolor}
\usepackage{setspace}

\titleformat{\section}{\large\bfseries}{}{0em}{}[\titlerule]
\titleformat{\subsection}{\bfseries}{}{0em}{}
\renewcommand{\labelitemii}{$\circ$}
\setlist[itemize]{noitemsep, topsep=0pt}
\setstretch{0.95}
\pagenumbering{gobble}

\begin{document}
\begin{center}
    {\LARGE \textbf{Suhas Oruganti}}\\
    \faPhone~408-598-5684 \quad
    \faEnvelope~\href{mailto:emailsuhas.o@gmail.com}{emailsuhas.o@gmail.com} \quad
    \faLinkedin~\href{https://www.linkedin.com/in/suhasoruganti/}{linkedin.com/in/suhasoruganti/} \quad
    \faGithub~\href{https://github.com/Suhas}{github.com/Suhas}\\
\end{center}

\section*{Education}
\textbf{University of California, Santa Cruz} \hfill \textbf{Expected Graduation: June 2026}\\
\textit{Bachelor of Science in Computer Science} \hfill \textbf{GPA: 4.0}\\
\textbf{Coursework:} Data Structures and Algorithms, Applied Machine Learning, Discrete Mathematics, Artificial Intelligence

\section*{Experience}
\textbf{Amazon} \hfill \textbf{June 2025 -- Present}\\
\textit{Software Development Engineer} \hfill \textit{Seattle, Washington}\\
\textbf{Incoming SDE Intern}

\textbf{Web3Names.ai} \hfill \textbf{September 2024 -- December 2024}\\
\textit{SWE + AI} \hfill \textit{San Francisco, California}\\
\begin{itemize}
    \item Conducted web scraping and data collection from decentralized platforms to build social graphs and analyze blockchain interactions.
    \item Developed custom Python scripts for automating outreach and engagement on Web3 social media platforms.
    \item Applied data science techniques to gather, process, and visualize Web3 user data for marketing and business intelligence.
\end{itemize}

\textbf{Tech4Good} \hfill \textbf{June 2024 -- Present}\\
\textit{Web Developer Intern} \hfill \textit{Santa Cruz, California}\\
\begin{itemize}
    \item Built a task management web app, Compass, to improve task efficiency for students using Angular.
    \item Implemented batch write services with asynchronous TypeScript for smooth goal creation and modification.
    \item Developed responsive, dynamic user interfaces with state management to ensure seamless goal tracking.
\end{itemize}

\textbf{AIEA Lab} \hfill \textbf{March 2022 -- Present}\\
\textit{Explainable AI for Autonomous Vehicles} \hfill \textit{Santa Cruz, California}\\
\begin{itemize}
    \item Developed reinforcement learning algorithms to enhance decision-making in autonomous vehicles using PPO.
    \item Simulated real-world driving scenarios in Duckietown, improving the accuracy of AI model decision-making.
    \item Integrated attention mechanisms for enhanced explainability of AI systems for regulators.
\end{itemize}

\section*{Projects}
\textbf{Meal-Connect} | \textit{CSS, HTML, Python, Typescript, Node.js, MongoDB, Express, Gemini API, Google Maps API}\\
\begin{itemize}
    \item Created a platform connecting restaurants with homeless shelters, reducing food waste through meal redistribution.
    \item Used Google Maps API to map food providers and shelters for efficient coordination.
    \item Implemented a chatbot using Google Gemini API for real-time insights on food waste and resource management.
\end{itemize}

\textbf{Button Alert} | \textit{Dart, Flutter, Google Cloud}\\
\begin{itemize}
    \item Developed a mobile app for emergency alerts with seamless functionality using Google Cloud.
    \item Expanded the app to a web-based platform, increasing user engagement by \textbf{15 Percent}.
\end{itemize}

\textbf{Grocery List} | \textit{Python, HTML, CSS, Flask, SQL}\\
\begin{itemize}
    \item Developed an online grocery store app, reducing ordering time by \textbf{20 percent} through optimized navigation and data handling.
    \item Managed user data and product inventories using SQL for better performance.
\end{itemize}

\section*{Technical Skills}
\textbf{Languages:} Python, Java, HTML, TypeScript, Dart, C, C++, CSS\\
\textbf{Technologies:} Angular, Firebase, Flask, Flutter, Google Cloud, Git, Node.js, React/ReactNative\\
\textbf{Developer Tools:} VS Code, GitHub, Docker

\section*{Leadership / Extracurricular}
\textbf{Associated Computing Machinery} \hfill \textbf{Fall 2023 -- Present}\\
\textit{Community Officer} \hfill \textit{Santa Cruz}\\
\begin{itemize}
    \item Organized various socials and workshops, engaging \textbf{200+ participants} and collaborating with industry professionals.
\end{itemize}
\end{document}

job description:""") + """{transcript}

Summary:"""
        )

        # Construct the chain
        chain = {"transcript": RunnablePassthrough()} | prompt | llm | StrOutputParser()

        # Run the chain to get a summary
        print(f"Generating summary using {model}...")
        summary = chain.invoke(job_description)

        return summary

    except requests.exceptions.ConnectionError:
        return f"Error: Could not connect to Ollama at {base_url}. Is Ollama running?"
    except requests.exceptions.Timeout:
        return f"Error: Request to Ollama timed out after {timeout} seconds"
    except requests.exceptions.RequestException as e:
        return f"Error connecting to Ollama: {str(e)}"
    except ValueError as e:
        return f"Error with Ollama configuration: {str(e)}"
    except Exception as e:  # pylint: disable=broad-exception-caught
        return f"Unexpected error during summarization: {str(e)}"


if __name__ == "__main__":
    print("\nGenerating summary, please wait...")

    job_description = escape_for_ollama(r"""
Looking for an opportunity with a dynamic, fun, and goal-oriented company? We’re growing quickly and we’re looking for some not-so-typical talent to join our team.

Progress Residential® is the largest providers of high-quality, single-family rental homes in the United States. With more than 90,000 homes across some of the fastest-growing markets, our residents appreciate the flexibility, freedom, and convenience of living in a single-family home without the obligations of home ownership.

Progress is committed to making the home rental process easy and enjoyable for the residents we serve by empowering our team members and investing in innovative systems and technology. Our portfolio has continued to grow substantially the past few years and we see increasing demand for professionally managed single-family rental homes and anticipate continued growth.

Employment with Progress Residential is conditional on a satisfactory background and drug screen.

 Text ProgressJobs to 25000 and you can chat with our Recruiting Assistant Kate who can help you find jobs, apply for jobs and answer your questions. 

Progress Residential is hiring a summer intern for our software engineering team. The Internship will provide a unique experience and invaluable look into our growing portfolio of single-family properties across the US and how we operate and manage these assets. The internship will accelerate your learning and development by providing you with a strong foundation upon which to understand the Single-Family Real Estate business as you develop your career. Program participants will receive real-world training and participate in networking and learning opportunities in an office environment.

We’re on a mission to hire the very best and are committed to creating exceptional employee experiences where everyone is respected and has access to equal opportunity. We realize that new ideas can come from everywhere in the organization, and we know the next big idea could be yours!

We are seeking a talented and passionate Full Stack Software Engineer Intern to extend your network and collaborate with engineers, architects, and leaders across the product engineering, and the business leadership teams!

The summer internship will begin in June 2025 and end in August 2025.

Essential Functions

Participating in all aspects of application development activities, including design, coding, code review, testing, bug fixing, and code/API documentation.
Grow with the support of your team and help others on the team grow by providing thoughtful feedback and uplifting those around you.
Work both independently and collaboratively within a fast-paced development team, with clear, positive, and constructive communication.
Additional responsibilities as needed based on specific role or team

Qualifications

Ability to travel locally to Progress office in Tempe, AZ throughout the program
Currently pursuing a Bachelor's or Master’s in Computer Science, Computer Engineering, Electrical Engineer, or equivalent experience required
Current student within an accredited college or university. 
Strong technical background with analytical and problem-solving skills
Software Engineering skills in Typescript, AWS Lamda, API connection and object database areas
Familiarity with front-end-frameworks such as React.js, node.js is a plus
Experience with building mobile app is a plus
Strong understanding of Devops pipelines, source control and cloud native technologies, as well as development of APIs for integrations by other teams
Customer focused and have real passion for quality and engineering excellence at scale.
Excellent communication and collaboration skills.
Experience with Agile methodologies
""")
    session_summary = get_ai_summary(job_description=job_description)

    print(f"\nSummary:\n{session_summary}")

def main():
    print("\nGenerating summary, please wait...")
    job_description = escape_for_ollama(r"""
[Your job description here]]
""")
    tailored_text = get_ai_summary(job_description=job_description)
    print(f"\nTailored LaTeX Code:\n{tailored_text}")

    #Attempting to compile LaTex code to PDF
    try:
        pdf_path = compile_latex_to_pdf(tailored_text)
        print(f"PDF generated at: {pdf_path}")
    except Exception as e:
        print(f"Error generating PDF: {e}")

if __name__ == "__main__":
    main()