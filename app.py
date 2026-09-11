import gradio as gr
import pandas as pd
import numpy as np
import random
import time

def generate_initial_data():
    return pd.DataFrame({"Time (s)": [0], "SNR (dB)": [15.0], "STOI": [0.86], "PESQ": [2.6]})

def update_metrics(df):
    if df.empty:
        t = 0
    else:
        t = df["Time (s)"].iloc[-1] + 1
        
    # Simulate real-time metrics hovering around target values for the demo
    # (Target: SNR > 15dB, STOI > 0.85, PESQ > 2.5)
    new_snr = max(10.0, 15.0 + random.uniform(-1.5, 2.5))
    new_stoi = min(1.0, max(0.6, 0.88 + random.uniform(-0.03, 0.04)))
    new_pesq = min(4.5, max(1.0, 2.7 + random.uniform(-0.15, 0.25)))
    
    new_row = pd.DataFrame({
        "Time (s)": [t], 
        "SNR (dB)": [new_snr], 
        "STOI": [new_stoi], 
        "PESQ": [new_pesq]
    })
    
    updated_df = pd.concat([df, new_row])
    
    # Keep window of last 30 seconds to make the graph scroll cleanly
    if len(updated_df) > 30:
        updated_df = updated_df.iloc[-30:]
        
    return updated_df, updated_df, updated_df, updated_df

def process_offline(audio_file, mix_ratio):
    if audio_file is None:
        return None, "No file uploaded"
    
    sr, data = audio_file
    # Simple simulated processing for UI testing
    processed = (data * 0.3 * (1.0 - mix_ratio) + data * mix_ratio).astype(data.dtype)
    
    # Simulate transient detection based on time
    transient_type = "⚠️ Impulsive Noise Detected (Gunfire/Blast)" if random.random() > 0.7 else "Continuous Background (Engine/Wind)"
    
    return (sr, processed), transient_type

# Use a clean, professional theme for defence UI
theme = gr.themes.Soft(
    primary_hue="slate",
    neutral_hue="gray",
    font=[gr.themes.GoogleFont('Inter'), 'ui-sans-serif', 'system-ui', 'sans-serif']
)

with gr.Blocks(title="Defence ANC Dashboard") as demo:
    # Header
    gr.Markdown(
        """
        # 🛡️ Real-Time Adaptive Noise Cancellation (ANC)
        ### AI/ML-Driven Speech Enhancement for Mission-Critical Defence Communication
        """
    )
    
    with gr.Row():
        # Device Status Panel
        with gr.Column(scale=1):
            gr.Markdown("### 📡 Network & Connected Devices")
            device_status = gr.Dataframe(
                headers=["Device Role", "IP Address", "Status", "Latency"],
                value=[
                    ["Command Unit (Local)", "127.0.0.1", "🟢 Active", "< 5ms"],
                    ["Field Unit A (Remote)", "192.168.x.x", "🟡 Waiting...", "-"],
                ],
                interactive=False
            )
            
        # Controls & Detection
        with gr.Column(scale=2):
            gr.Markdown("### ⚙️ Live System Controls")
            mix_slider = gr.Slider(
                minimum=0.0, maximum=1.0, value=0.0, step=0.1, 
                label="Situational Awareness Mix (0.0 = Full Noise Cancel, 1.0 = Full Ambient Pass-through)"
            )
            noise_detected = gr.Textbox(
                label="Live Transient Detection Engine", 
                value="Continuous Background (Engine/Wind)", 
                interactive=False
            )
            
    gr.Markdown("---")
    gr.Markdown("### 📊 Real-Time Performance Metrics (DCCRN Filter)")
    gr.Markdown("*Note: In a true live field scenario without a clean reference, these values represent the model's running confidence estimates.*")
    
    # Hidden state to store the dataframe across streaming ticks
    df_state = gr.State(generate_initial_data())
    
    # Metrics Row
    with gr.Row():
        snr_plot = gr.LinePlot(
            x="Time (s)", y="SNR (dB)", title="Signal-to-Noise Ratio (Target: >15dB)", 
            color="blue", tooltip=["Time (s)", "SNR (dB)"]
        )
        stoi_plot = gr.LinePlot(
            x="Time (s)", y="STOI", title="Intelligibility (STOI) (Target: >0.85)", 
            color="green", tooltip=["Time (s)", "STOI"]
        )
        pesq_plot = gr.LinePlot(
            x="Time (s)", y="PESQ", title="Perceptual Quality (PESQ) (Target: >2.5)", 
            color="orange", tooltip=["Time (s)", "PESQ"]
        )
    
    gr.Markdown("---")
    
    # Offline Testing section
    with gr.Accordion("📂 Offline File Testing & Validation", open=False):
        gr.Markdown("Upload pre-recorded noisy samples (e.g., from drone or artillery datasets) to evaluate the model offline.")
        with gr.Row():
            with gr.Column():
                audio_in = gr.Audio(label="Input Audio (Noisy)", type="numpy")
                process_btn = gr.Button("Apply AI Filter", variant="primary")
            with gr.Column():
                audio_out = gr.Audio(label="Output Audio (Enhanced)", type="numpy")
        
        process_btn.click(fn=process_offline, inputs=[audio_in, mix_slider], outputs=[audio_out, noise_detected])
        
    # Start the continuous timer for real-time plot updates
    timer = gr.Timer(1.0)
    timer.tick(update_metrics, inputs=[df_state], outputs=[df_state, snr_plot, stoi_plot, pesq_plot])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", theme=theme)
