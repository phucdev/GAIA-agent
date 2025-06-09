# import inspect
import json
import os
from pathlib import Path
from typing import Dict
from zipfile import ZipFile

import gradio as gr
import pandas as pd
import requests

from agent import BasicAgent

# (Keep Constants as is)
# --- Constants ---
DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"

with open("prompt.json", mode="r") as f:
    prompt_template = json.load(f)


def post_process_answer(answer: str) -> str:
    """Post-process the answer to extract the final answer."""
    if "FINAL ANSWER:" in answer:
        answer = answer.split("FINAL ANSWER:")[-1].strip()
    return answer


def solve_question(question: Dict[str, str]) -> Dict[str, str]:
    """Solve the question using the BasicAgent."""
    agent = BasicAgent()
    question_text = question.get("question")
    task_id = question.get("task_id")
    if not question_text:
        raise ValueError("Question text is empty.")
    augmented_question = prompt_template["user_prompt"] + question_text
    if question.get("file_name"):
        file_url = DEFAULT_API_URL + "/files"
        response = requests.get(f"{file_url}/{question['task_id']}", timeout=15)
        # Check if the request was successful
        if response.status_code != 200:
            raise ValueError(f"Failed to fetch file for task {task_id}: {response.status_code} - {response.text}")
        file_path = Path("files") / question["file_name"]
        # Create files directory if it doesn't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(response.content)
        if file_path.suffix == "zip":
            # If the file is a zip, we need to extract the files and give the LLM the list of files
            file_paths = []
            with ZipFile(file_path, "r") as zip_ref:
                for file_info in zip_ref.infolist():
                    # Read file content
                    file_data = zip_ref.read(file_info.filename)
                    extracted_file_path = file_path / file_info.filename
                    with open(extracted_file_path, "wb") as extracted_file:
                        extracted_file.write(file_data)
                    file_paths.append(str(extracted_file_path))
            augmented_question += prompt_template["use_files_prompt"] + str(file_paths)
        else:
            augmented_question += prompt_template["use_file_prompt"] + str(file_path)
    try:
        agent_response = agent(augmented_question)
        submitted_answer = post_process_answer(agent_response)
        return {
            "Task ID": task_id,
            "Question": augmented_question,
            "Submitted Answer": submitted_answer,
            "Full Answer": agent_response,
        }
    except Exception as e:
        print(f"Error running agent on task {task_id}: {e}")
        return {
            "Task ID": task_id,
            "Question": augmented_question,
            "Submitted Answer": f"AGENT ERROR: {e}",
            "Full Answer": "",
        }


def run_and_submit_all(profile: gr.OAuthProfile | None):
    """Fetches all questions, runs the BasicAgent on them, submits all answers, and displays the
    results."""
    # --- Determine HF Space Runtime URL and Repo URL ---
    # Get the SPACE_ID for sending link to the code
    space_id = os.getenv("SPACE_ID")

    if profile:
        username = f"{profile.username}"
        print(f"User logged in: {username}")
    else:
        print("User not logged in.")
        return "Please Login to Hugging Face with the button.", None

    api_url = DEFAULT_API_URL
    questions_url = f"{api_url}/questions"
    submit_url = f"{api_url}/submit"

    # In the case of an app running as a hugging Face space, this link points
    # toward your codebase ( useful for others so please keep it public)
    agent_code = f"https://huggingface.co/spaces/{space_id}/tree/main"
    print(agent_code)

    # Fetch Questions
    print(f"Fetching questions from: {questions_url}")
    try:
        response = requests.get(questions_url, timeout=15)
        response.raise_for_status()
        questions_data = response.json()
        if not questions_data:
            print("Fetched questions list is empty.")
            return "Fetched questions list is empty or invalid format.", None
        print(f"Fetched {len(questions_data)} questions.")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching questions: {e}")
        return f"Error fetching questions: {e}", None
    except requests.exceptions.JSONDecodeError as e:
        print(f"Error decoding JSON response from questions endpoint: {e}")
        print(f"Response text: {response.text[:500]}")
        return f"Error decoding server response for questions: {e}", None
    except Exception as e:
        print(f"An unexpected error occurred fetching questions: {e}")
        return f"An unexpected error occurred fetching questions: {e}", None

    # Run your Agent
    results_log = []
    answers_payload = []

    results_file_path = Path("files/results_log.jsonl")
    results_file_path.parent.mkdir(parents=True, exist_ok=True)
    solved_task_ids = []
    if results_file_path.exists():
        print(f"Results file already exists: {results_file_path}")
        with open(results_file_path, "r") as results_file:
            for line in results_file:
                result = json.loads(line)
                results_log.append(result)
                solved_task_ids.append(result["Task ID"])
    filtered_questions_data = [
        question
        for question in questions_data
        if question["task_id"] not in solved_task_ids
    ]
    if solved_task_ids:
        print(
            f"Found {len(solved_task_ids)} solved questions. "
            f"Running agent on remaining {len(filtered_questions_data)} questions."
        )
    else:
        print(f"Running agent on {len(questions_data)} questions...")
    for item in filtered_questions_data:
        result = solve_question(item)
        results_log.append(result)
    with open(results_file_path, "w") as results_file:
        for result in results_log:
            results_file.write(json.dumps(result) + "\n")
    for result in results_log:
        answers_payload.append(
            {
                "task_id": result["Task ID"],
                "submitted_answer": result["Submitted Answer"],
            }
        )

    if not answers_payload:
        print("Agent did not produce any answers to submit.")
        return (
            "Agent did not produce any answers to submit.",
            pd.DataFrame(results_log),
        )

    # 4. Prepare Submission
    submission_data = {
        "username": username.strip(),
        "agent_code": agent_code,
        "answers": answers_payload,
    }
    status_update = (
        f"Agent finished. Submitting {len(answers_payload)} "
        f"answers for user '{username}'..."
    )
    print(status_update)

    # 5. Submit
    print(f"Submitting {len(answers_payload)} answers to: {submit_url}")
    try:
        response = requests.post(submit_url, json=submission_data, timeout=60)
        response.raise_for_status()
        result_data = response.json()
        final_status = (
            f"Submission Successful!\n"
            f"User: {result_data.get('username')}\n"
            f"Overall Score: {result_data.get('score', 'N/A')}% "
            f"({result_data.get('correct_count', '?')}/"
            f"{result_data.get('total_attempted', '?')} correct)\n"
            f"Message: {result_data.get('message', 'No message received.')}"
        )
        print("Submission successful.")
        results_df = pd.DataFrame(results_log)
        return final_status, results_df
    except requests.exceptions.HTTPError as e:
        error_detail = f"Server responded with status {e.response.status_code}."
        try:
            error_json = e.response.json()
            error_detail += f" Detail: {error_json.get('detail', e.response.text)}"
        except requests.exceptions.JSONDecodeError:
            error_detail += f" Response: {e.response.text[:500]}"
        status_message = f"Submission Failed: {error_detail}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except requests.exceptions.Timeout:
        status_message = "Submission Failed: The request timed out."
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except requests.exceptions.RequestException as e:
        status_message = f"Submission Failed: Network error - {e}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except Exception as e:
        status_message = f"An unexpected error occurred during submission: {e}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df


# --- Build Gradio Interface using Blocks ---
with gr.Blocks() as demo:
    gr.Markdown("# Basic Agent Evaluation Runner")
    gr.Markdown(
        """
        **Instructions:**

        1.  Please clone this space, then modify the code to define your agent's logic, the tools, the necessary packages, etc ...
        2.  Log in to your Hugging Face account using the button below. This uses your HF username for submission.
        3.  Click 'Run Evaluation & Submit All Answers' to fetch questions, run your agent, submit answers, and see the score.

        ---
        **Disclaimers:**
        Once clicking on the "submit button, it can take quite some time ( this is the time for the agent to go through all the questions).
        This space provides a basic setup and is intentionally sub-optimal to encourage you to develop your own, more robust solution. For instance for the delay process of the submit button, a solution could be to cache the answers and submit in a separate action or even to answer the questions in async.
        """
    )

    gr.LoginButton()

    run_button = gr.Button("Run Evaluation & Submit All Answers")

    status_output = gr.Textbox(
        label="Run Status / Submission Result", lines=5, interactive=False
    )
    # Removed max_rows=10 from DataFrame constructor
    results_table = gr.DataFrame(label="Questions and Agent Answers", wrap=True)

    run_button.click(fn=run_and_submit_all, outputs=[status_output, results_table])

if __name__ == "__main__":
    print("\n" + "-" * 30 + " App Starting " + "-" * 30)
    # Check for SPACE_HOST and SPACE_ID at startup for information
    space_host_startup = os.getenv("SPACE_HOST")
    space_id_startup = os.getenv("SPACE_ID")  # Get SPACE_ID at startup

    if space_host_startup:
        print(f"✅ SPACE_HOST found: {space_host_startup}")
        print(f"   Runtime URL should be: https://{space_host_startup}.hf.space")
    else:
        print("ℹ️  SPACE_HOST environment variable not found (running locally?).")

    if space_id_startup:  # Print repo URLs if SPACE_ID is found
        print(f"✅ SPACE_ID found: {space_id_startup}")
        print(f"   Repo URL: https://huggingface.co/spaces/{space_id_startup}")
        print(
            f"   Repo Tree URL: https://huggingface.co/spaces/"
            f"{space_id_startup}/tree/main"
        )
    else:
        print(
            "ℹ️  SPACE_ID environment variable not found (running locally?). "
            "Repo URL cannot be determined."
        )

    print("-" * (60 + len(" App Starting ")) + "\n")

    print("Launching Gradio Interface for Basic Agent Evaluation...")
    demo.launch(debug=True, share=False)
