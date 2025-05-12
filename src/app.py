import streamlit as st
import pandas as pd
from inputs import get_user_inputs
import utils.st_utils as st_utils

# Constants
VALID_TOKEN_KEY = "valid_token"
OPENAI_TOKEN_KEY = "openai_token"
SEQUENCE_INPUT_KEY = "sequence_input"
PDB_STRUCTURES_KEY = "pdb_structures"
EVALUATION_RESULTS_KEY = "evaluation_results"
FOXM1 = "FOXM1"
MYC = "MYC"
RMSD = "RMSD"

# Streamlit UI
st.title("Protein Structure Prediction using MC-NEST")

st.sidebar.header("API Configuration")
st.session_state[VALID_TOKEN_KEY] = st.session_state.get(VALID_TOKEN_KEY, False)
st.session_state[OPENAI_TOKEN_KEY] = st.session_state.get(OPENAI_TOKEN_KEY, None)
st.session_state[PDB_STRUCTURES_KEY] = st.session_state.get(PDB_STRUCTURES_KEY, [])
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
        st.success(best_hypothesis)
        
        
        # Create a DataFrame for easy visualization in paper
        df = pd.DataFrame(st.session_state[EVALUATION_RESULTS_KEY]).T
        st.markdown("### Protein Evaluation Metrics")
        st.dataframe(df, use_container_width=True)
        st.info(f"RMSD between FOXM1 and MYC structures: {st.session_state[EVALUATION_RESULTS_KEY][MYC][RMSD]:.3f} Å")
    else:
        st.warning("Please enter a valid protein sequence.")

# --- Footer: Supported Organization Logos ---
st_utils.show_organization_logos()