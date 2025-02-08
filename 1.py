import os
import time
import tempfile
from pathlib import Path
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.duckduckgo import DuckDuckGo
from google.generativeai import upload_file, get_file
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if API_KEY:
    genai.configure(api_key=API_KEY)

# Initialize AI agent
def initialize_agent():
    return Agent(
        name="Video AI Summarizer",
        model=Gemini(id="gemini-2.0-flash-exp"),
        tools=[DuckDuckGo()],
        markdown=True,
    )

multimodal_Agent = initialize_agent()

def analyze_video(video_path, user_query):
    try:
        print("Processing video...")
        processed_video = upload_file(video_path)
        print(processed_video)
        while processed_video.state.name == "PROCESSING":
            time.sleep(1)
            processed_video = get_file(processed_video.name)

        analysis_prompt = (
            f"""
            Extract detailed insights from the uploaded video, structuring the summary for student notes.
            The summary should include:
            
            1. **Key Topics Covered**
            2. **Important Definitions & Terminologies**
            3. **Step-by-Step Explanation of Concepts**
            4. **Real-World Applications & Examples**
            5. **Critical Insights & Takeaways**
            6. **Additional Supporting Information from Web Research**
            
            Ensure the summary is detailed, well-structured, and easy to understand, making it useful for study purposes.
            
            **User Query:** {user_query}
            """
        )

        response = multimodal_Agent.run(analysis_prompt, videos=[processed_video])
        summary = response.content

        # Save summary in a new folder
        output_folder = "video_summaries"
        os.makedirs(output_folder, exist_ok=True)
        summary_file = os.path.join(output_folder, f"{Path(video_path).stem}_summary.txt")
        
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write(summary)
        
        print(f"Summary saved to {summary_file}")
        return summary
    
    except Exception as error:
        print(f"An error occurred: {error}")
    finally:
        Path(video_path).unlink(missing_ok=True)

if __name__ == "__main__":
    video_path = input("Enter the path of the video file: ")
    if not os.path.exists(video_path):
        print("Invalid file path. Please check and try again.")
    else:
        user_query = input("What insights are you seeking from the video? ")
        summary = analyze_video(video_path, user_query)
        if summary:
            print("\nVideo Analysis Summary:")
            print(summary)
