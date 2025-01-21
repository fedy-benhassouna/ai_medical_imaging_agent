import os
import numpy as np
import gradio as gr
from PIL import Image
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.duckduckgo import DuckDuckGo
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Medical Analysis Query - Reformatted for better processing
ANALYSIS_QUERY = """Analyze this medical image as a highly skilled medical imaging expert with extensive knowledge in radiology and diagnostic imaging. Follow this structure:

1. Image Type & Region Analysis:
* Identify the imaging modality used (X-ray/MRI/CT/Ultrasound/etc.)
* Describe the anatomical region shown and patient positioning
* Assess image quality and technical adequacy

2. Key Findings Analysis:
* Describe all primary observations in detail
* Document any abnormalities with precise descriptions
* Note relevant measurements and densities
* Specify locations, sizes, shapes, and key characteristics
* Indicate severity level (Normal/Mild/Moderate/Severe)

3. Diagnostic Assessment:
* State your primary diagnosis and confidence level
* List potential differential diagnoses in order of likelihood
* Provide evidence from the image supporting each diagnosis
* Highlight any urgent or critical findings requiring immediate attention

4. Patient-Friendly Summary:
* Explain the findings in clear, simple language
* Define any necessary medical terms
* Use helpful analogies where appropriate
* Address likely patient concerns

5. Research Context:
* Search recent medical literature for similar cases using DuckDuckGo
* Find relevant treatment protocols
* Identify recent technological advances in this area
* Provide 2-3 key medical references supporting your analysis

Please provide a comprehensive analysis covering ALL sections above."""

# Initialize the medical agent
medical_agent = Agent(
    model=Gemini(
        api_key=os.getenv('GOOGLE_API_KEY'),
        id="gemini-2.0-flash-exp"
    ),
    tools=[DuckDuckGo()],
    markdown=True
)

def analyze_medical_image(image):
    """
    Analyze the uploaded medical image using the medical agent.
    """
    if image is None:
        gr.Warning("Please upload an image first")
        return "Please upload an image to begin analysis."
    
    try:
        # Save the image temporarily
        temp_path = "temp_medical_image.png"
        
        # Handle different image input types from Gradio
        if isinstance(image, np.ndarray):
            # Convert numpy array to PIL Image
            img = Image.fromarray(image)
        elif isinstance(image, Image.Image):
            img = image
        else:
            gr.Warning("Invalid image format")
            return "Invalid image format"
            
        # Save image and ensure it's in RGB mode
        img = img.convert('RGB')
        img.save(temp_path)
        
        gr.Info("Analysis in progress... Please wait.")
        
        # Run analysis with explicit instructions
        with open(temp_path, 'rb') as img_file:
            response = medical_agent.run(
                ANALYSIS_QUERY + "\n\nIMPORTANT: Please ensure you address ALL sections (1-5) in your analysis.",
                images=[temp_path]
            )
        
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        # Check if response seems incomplete
        if len(response.content.strip()) < 100:  # Basic length check
            gr.Warning("Analysis seems incomplete. Please try again.")
            return "Error: Analysis seems incomplete. Please try again."
            
        gr.Success("Analysis completed successfully!")
        return response.content
        
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        gr.Error(f"Error during analysis: {str(e)}")
        return f"Error during analysis: {str(e)}"

# Create the Gradio interface
with gr.Blocks(title="Medical Imaging Diagnosis Agent") as demo:
    gr.Markdown("""
    # 🏥 Medical Imaging Diagnosis Agent
    Upload a medical image for professional analysis
    
    ⚠️ **DISCLAIMER**: This tool is for educational and informational purposes only. 
    All analyses should be reviewed by qualified healthcare professionals. 
    Do not make medical decisions based solely on this analysis.
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(
                label="Upload Medical Image",
                sources=["upload"],
                type="pil"
            )
            analyze_button = gr.Button(
                "🔍 Analyze Image",
                variant="primary"
            )
        
        with gr.Column(scale=1):
            output_text = gr.Markdown(
                label="Analysis Results",
                value="Upload an image and click 'Analyze Image' to begin."
            )
    
    gr.Markdown("""
    ℹ️ **Tool Information**:
    This tool provides AI-powered analysis of medical imaging data using advanced computer vision and radiological expertise.
    """)
    
    # Set up the click event
    analyze_button.click(
        fn=analyze_medical_image,
        inputs=[input_image],
        outputs=[output_text]
    )

# Launch the app
if __name__ == "__main__":
    demo.launch()