import streamlit as st
import pandas as pd
from inputs import get_user_inputs
import utils.st_utils as st_utils
from utils.pdf_report_generate import run_full_pdf_generation_ui
from datetime import datetime, timezone
import uuid
from utils.constants import (
    VALID_TOKEN_KEY,
    OPENAI_TOKEN_KEY,
    SEQUENCE_INPUT_KEY,
    PDB_STRUCTURES_KEY,
    BEST_HYPOTHESIS_KEY,
    EVALUATION_RESULTS_KEY,
    FOXM1,
    MYC,
    RMSD
)
# -------------------- Streamlit UI for Protein Structure Prediction -------------------- #



if "session_id" not in st.session_state:
    timestamp = datetime.now(timezone.utc).isoformat()
    st.session_state.session_id = f"{uuid.uuid4()}_{timestamp}"
    
# Streamlit UI
st.title("AI as Co-Scientist: Collaborative AI Agents")

st.sidebar.header("API Configuration")
st.session_state[VALID_TOKEN_KEY] = st.session_state.get(VALID_TOKEN_KEY, False)
st.session_state[OPENAI_TOKEN_KEY] = st.session_state.get(OPENAI_TOKEN_KEY, None)
st.session_state[PDB_STRUCTURES_KEY] = st.session_state.get(PDB_STRUCTURES_KEY, [])
st.session_state[BEST_HYPOTHESIS_KEY] = st.session_state.get(BEST_HYPOTHESIS_KEY, None)
st.session_state[EVALUATION_RESULTS_KEY] = st.session_state.get(EVALUATION_RESULTS_KEY, {})
# Load all input values
user_config = get_user_inputs()

def process_user_input(sequence_input: str)-> None:
    pdb = st_utils.show_pdb_structure(
        title="User Input Structure",
        sequence=sequence_input
    )
    if pdb:
        st.session_state[PDB_STRUCTURES_KEY].append(pdb)
        st.session_state[EVALUATION_RESULTS_KEY][FOXM1] = st_utils.get_bio_prop_evaluation_results(
            sequence_input
        )

def process_synthetic_structure(user_config: dict) -> str:
    best_hypothesis, myc = st_utils.get_predicted_sequence_with_hypothesis(user_config)
    pdb = st_utils.show_pdb_structure(
        title="Synthetic Structure",
        sequence=myc
    )
    if pdb:
        st.session_state[PDB_STRUCTURES_KEY].append(pdb)
        st.session_state[EVALUATION_RESULTS_KEY][MYC] = st_utils.get_bio_prop_evaluation_results(
            sequence=myc
        )
    return best_hypothesis

# Main logic
if st.session_state.valid_token and st.button("Predict & Visualize"):
    if user_config[SEQUENCE_INPUT_KEY].strip():
        col1, col2 = st.columns(2)
        with col1:
            process_user_input(user_config[SEQUENCE_INPUT_KEY].strip())
        with col2:
            with st.spinner("Loading data in..."):
                best_hypothesis = process_synthetic_structure(user_config)
        
        # Display the predicted structure and evaluation results
        st.session_state[BEST_HYPOTHESIS_KEY] = best_hypothesis
        st.success(best_hypothesis)
        
        
        # Create a DataFrame for easy visualization in paper
        df = pd.DataFrame(st.session_state[EVALUATION_RESULTS_KEY]).T
        st.markdown("### Protein Evaluation Metrics")
        st.dataframe(df, use_container_width=True)
        st.info(f"RMSD between FOXM1 and MYC structures: {st.session_state[EVALUATION_RESULTS_KEY][MYC][RMSD]:.3f} Å")
    else:
        st.warning("Please enter a valid protein sequence.")

if all([st.session_state[OPENAI_TOKEN_KEY], st.session_state[BEST_HYPOTHESIS_KEY]]):
    run_full_pdf_generation_ui(
        st.session_state[OPENAI_TOKEN_KEY],
        st.session_state[BEST_HYPOTHESIS_KEY],
        # st.session_state[EVALUATION_RESULTS_KEY],
        # st.session_state[PDB_STRUCTURES_KEY][0]
    )
        
# --- Footer: Supported Organization Logos ---
# st_utils.show_organization_logos()