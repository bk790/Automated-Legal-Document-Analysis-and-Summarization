import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from pdf2image import convert_from_bytes
import numpy as np
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from math import ceil

@st.cache_resource
def load_data():
    # Load the pre-trained LayoutLMv3 model
    model = LayoutLMv3ForTokenClassification.from_pretrained("/home/bhanu/ji/data/Model")
    # Load the processor that prepares images and text for the LayoutLMv3 model
    processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base")
    tags = [
    "0", "Reference Document Number", "Provision Of Law", "Annexure Number","Annexure Document Description","Fact Sequence","Respondent 2 Address","Respondent 2 Age","Case Type",
    "Ground Sequence", "Petitioner Name", "Petitioner Age", "Petitioner Relation","Petitioner Address","Respondent 1","Respondent 2 Name","Respondent 2 Relation","Case Heading",
    "Statement Of Facts Sequence","Challenged Document","Petitioner","Respondent"
]
    # Create mappings for labels to numeric IDs 
    ids_to_tags = {k:v for k,v in enumerate(tags)}
    tag_labels = list(ids_to_tags.keys())[1:]
    return model,processor,ids_to_tags,tag_labels

# Load the model, processor, and tag mappings
model,processor,ids_to_tags,tag_labels = load_data()

# Helper function to split continuous word indices for the same line or paragraph.
def split_continuous(numbers):
    result = []
    if numbers[0].size==0:
        return result
    else:
        pg_index = numbers[0]
        word_index = numbers[1]
        current_sublist = [pg_index[0],word_index[0]]

        # Iterate through word indices to split them into continuous blocks
        for i in range(1, len(word_index)):
            if word_index[i] == word_index[i-1] + 1:
                current_sublist.append(word_index[i])
            else:
                result.append(current_sublist)
                current_sublist = [pg_index[i],word_index[i]]

        result.append(current_sublist)
        return result

@st.cache_data(show_spinner="Processing....")
def predict(pdf_file):
    # Convert PDF to images
    images = convert_from_bytes(pdf_file)
    # Use processor to tokenize the images and extract bounding boxes
    encoded_data = processor(images,return_tensors="pt",max_length=512,padding="max_length",truncation=True)
    # Extract token IDs and bounding boxes
    token_ids = encoded_data.input_ids
    bboxes = encoded_data.bbox
    out = model(**encoded_data)

    #Get predicted tags (the label IDs with highest logits)
    predicted_tags = np.array(out.logits.argmax(-1))

    # Check if any relevant tags were predicted (i.e., > 0)
    if (predicted_tags>0).any():
        # If all tags are found, validate as "Judgment/Orders"
        if np.unique(predicted_tags).size > 7:
            st.success("Successfully Validated : Statement Of Facts",icon="✅")
        else:
            # Warn about missing key data if fewer than 7 tags are found
            warn,missing_list = st.columns([3,1])
            warn.warning("Incomplete : Key Data Missing",icon="⚠️")
            missing = np.setdiff1d(tag_labels,np.unique(predicted_tags))
            pop = missing_list.popover("Missing Data")
            for i in missing:
                pop.error(ids_to_tags[i])

        # Organize the predicted tag indices and split them into continuous blocks
        tags_indices = {num: np.where(predicted_tags == num) for num in tag_labels}
        updated_tags_indices = dict(map(lambda item: (item[0], split_continuous(item[1])),tags_indices.items()))

        # Set font size for drawing on the images
        font = ImageFont.load_default(size=40)
        img_width, img_height = images[0].size
        w1_scale = img_width/1000
        h1_scale = img_height/1000

        # Draw bounding boxes and tags on the images
        for tag,indices in updated_tags_indices.items():
            for i in indices:
                draw = ImageDraw.Draw(images[i[0]])
                coordinates = bboxes[i[0]][i[1:]].tolist()
                left_bottom = (min(coord[0] for coord in coordinates)*w1_scale, min(coord[1] for coord in coordinates)*h1_scale)
                top_right = (max(coord[2] for coord in coordinates)*w1_scale, max(coord[3] for coord in coordinates)*h1_scale)
                draw.rectangle([left_bottom,top_right],outline="red",width=2)
                draw.text((left_bottom[0]+(top_right[0]-left_bottom[0])/2,left_bottom[1]-50),ids_to_tags[tag],fill="blue",font=font)

        # Display extracted data in an expandable section
        with st.expander("Extracted Data",icon="📊"):
            for tag,indices in updated_tags_indices.items():
                result = [processor.tokenizer.decode(token_ids[i[0]][i[1:]]) for i in indices] if len(indices)>0 else "None"
                color="red" if result=="None" else "green"
                st.write(f":blue[{ids_to_tags[tag]}] : :{color}[{' | '.join(result) if tag >3 else result[0] if isinstance(result,list) else ' '.join(result)}]")
         
        # Display the images with annotations
        for i,img in enumerate(images):
            with st.expander(f"Page {i+1}"):
                st.image(img)

    # If validation fails, display an error message
    else:
        st.error("Validation Failed : Please uplaod Statement Of Facts",icon="❌")

st.title("Document :blue[AI]",anchor=False)
# Upload the PDF file
uploaded_file = st.file_uploader("Upload file",type=".pdf",label_visibility="collapsed")

# If the file is uploaded, call the predict function
if uploaded_file:
    predict(uploaded_file.read())
