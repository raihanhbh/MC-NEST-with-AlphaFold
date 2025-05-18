import streamlit as st
import utils.st_utils as st_utils

def get_user_inputs():
    openai_token = st.sidebar.text_input(
        label="Enter OpenAI API Token",
        type="password",
        help="Paste your OpenAI token here. It will be used for secure API calls."
    )
    
    match(st_utils.is_valid_token(openai_token, st)):
        case True:
            st.sidebar.success("Token is valid and ready for use......  ")
        case False:
            st.sidebar.error("Invalid OpenAI API token. Please check and try again.")
        case None:
            st.sidebar.warning("Please enter your OpenAI API key to proceed located in the sidebar.")

    
    default_sequence = "MARTKQTARKSTGGKAPRKQLASKAARKSAARAAAAGGGGGGG"
    sequence_input = st.text_input("Enter a protein sequence (e.g., FOXM1) to predict and visualize its structure.", value=default_sequence, max_chars=1000, help="Enter the protein sequence to predict its structure.")

    zero_shot = 1
    # zero_shot = st.sidebar.number_input(
    #     label="Zero Shot",
    #     min_value=1,
    #     max_value=1,
    #     value=1,
    #     step=1,
    #     help="Controls how many Zero Shots are done. Higher values increase OpenAI API cost."
    # )

    max_rollouts = st.sidebar.number_input(
        label="Max Rollouts",
        min_value=1,
        max_value=10,
        value=2,
        step=1,
        help="Controls how many rollouts are done. Higher values increase OpenAI API cost."
    )

    # Mapping dictionary
    policy_map = {
        "GREEDY": 1,
        "IMPORTANCE_SAMPLING": 2,
        "PAIRWISE_IMPORTANCE_SAMPLING": 3
    }

    # Get list of options from the dictionary
    options = list(policy_map.keys())

    # Set default index to IMPORTANCE_SAMPLING
    default_index = options.index("IMPORTANCE_SAMPLING")

    selection_policy_label = st.sidebar.radio(
        label="Selection Policy",
        options=list(policy_map.keys()),
        index=default_index,
        help="Choose how the model selects rollouts for evaluation."
    )
    
    # Get the mapped value
    selection_policy_value = policy_map[selection_policy_label]
    
    # Return all input values
    return {
        "openai_token": openai_token or None,
        "sequence_input": sequence_input,
        "zero_shot": zero_shot,
        "max_rollouts": max_rollouts,
        "selection_policy_value": selection_policy_value
    }