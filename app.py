import gradio as gr
import subprocess
import os
import pandas as pd

def run_ranking():
    try:
        # We assume sample_candidates.json is inside India_runs_data_and_ai_challenge
        sample_path = "India_runs_data_and_ai_challenge/sample_candidates.json"
        out_csv = "India_runs_data_and_ai_challenge/submission.csv"
        
        if not os.path.exists(sample_path):
            return None, f"Error: {sample_path} not found. Make sure all repo files are uploaded."
            
        # 1. Precompute features and embeddings
        print("Running precompute...")
        p1 = subprocess.run(
            ["python", "precompute.py", "--candidates", sample_path], 
            capture_output=True, text=True
        )
        if p1.returncode != 0:
            return None, f"Precompute Error:\n{p1.stderr}"
            
        # 2. Rank candidates
        print("Running ranker...")
        p2 = subprocess.run(
            ["python", "rank.py", "--candidates", sample_path, "--out", out_csv],
            capture_output=True, text=True
        )
        if p2.returncode != 0:
            return None, f"Rank Error:\n{p2.stderr}"
            
        # 3. Load top 10 for preview
        if os.path.exists(out_csv):
            df = pd.read_csv(out_csv)
            preview = df.head(10)
            return out_csv, preview
        else:
            return None, "Error: Output CSV was not generated."
            
    except Exception as e:
        return None, f"An unexpected error occurred: {str(e)}"

# Define the Gradio interface
with gr.Blocks(title="Redrob Ranker - Bitwise Developers") as demo:
    gr.Markdown("# 🚀 Redrob Ranker Sandbox")
    gr.Markdown("Click the button below to run the hybrid semantic + heuristic ranking pipeline on the sample candidate dataset (100 candidates).")
    
    with gr.Row():
        run_btn = gr.Button("Run Ranking Pipeline", variant="primary")
        
    with gr.Row():
        file_output = gr.File(label="Download Full submission.csv")
    
    with gr.Row():
        df_output = gr.Dataframe(label="Top 10 Candidates Preview")
        
    run_btn.click(
        fn=run_ranking,
        inputs=[],
        outputs=[file_output, df_output]
    )

if __name__ == "__main__":
    demo.launch()
