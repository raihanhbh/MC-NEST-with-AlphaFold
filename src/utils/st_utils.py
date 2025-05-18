import streamlit as st
import streamlit.components.v1 as components
import requests
from Bio.PDB import PDBParser, Superimposer
from io import StringIO, BytesIO
import py3Dmol
from PIL import Image
import re
import os
import tempfile
import matplotlib.pyplot as plt
from predict_proteins import MC_NEST_gpt4o

VALID_TOKEN_KEY = "valid_token"
OPENAI_TOKEN_KEY = "openai_token"
SEQUENCE_INPUT_KEY = "sequence_input"
PDB_STRUCTURES_KEY = "pdb_structures"
EVALUATION_RESULTS_KEY = "evaluation_results"

# ESMFold prediction API
def predict_structure(sequence: str)-> str:
    url = "https://api.esmatlas.com/foldSequence/v1/pdb/"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    response = requests.post(url, data=sequence, headers=headers)
    if response.status_code == 200:
        return response.text
    else:
        st.error(f"Failed to predict structure: {response.status_code}")
        return None

def show_structure(pdb_data: dict, name: str)-> None:
    st.subheader(f"{name}")
    view = py3Dmol.view(width=350, height=400)
    view.addModel(pdb_data, "pdb")
    view.setStyle({"cartoon": {"color": "spectrum"}})
    view.zoomTo()
    view.setBackgroundColor("white")
    html = view._make_html()
    components.html(html, height=400)

def plot_pdb_backbone(pdb_data: str, output_path: str):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdb", mode='w') as f:
        f.write(pdb_data)
        file_path = f.name

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("X", file_path)

    x, y, z = [], [], []
    for atom in structure.get_atoms():
        if atom.get_name() == "CA":  # alpha-carbon backbone
            coord = atom.get_coord()
            x.append(coord[0])
            y.append(coord[1])
            z.append(coord[2])

    plt.figure(figsize=(6, 4))
    plt.plot(x, y, '-o')
    plt.title("Backbone (CA) 2D Projection")
    plt.savefig(output_path)
    plt.close()

    os.remove(file_path)

def save_pdb_as_png_with_pymol(pdb_data: str, output_path: str):
    # Write PDB to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdb") as temp_pdb:
        temp_pdb.write(pdb_data.encode())
        temp_pdb_path = temp_pdb.name

    print(f"Temp PDB file created at: {temp_pdb_path}")
    # PyMOL rendering script
    pymol_script = f"""
    load {temp_pdb_path}
    hide everything
    show cartoon
    spectrum count, rainbow
    bg_color white
    png {output_path}, width=800, height=600, ray=1
    quit
    """
    print(f"PyMOL script created:\n{pymol_script}")

    # Write script to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pml") as temp_script:
        temp_script.write(pymol_script.encode())
        temp_script_path = temp_script.name

    # Execute PyMOL
    os.system(f"pymol -cq {temp_script_path}")

    # Clean up
    # os.unlink(temp_pdb_path)
    # os.unlink(temp_script_path)

def save_pdb_as_png(pdb_data: str, output_path: str = "structure.png"):
    view = py3Dmol.view(width=400, height=400)
    view.addModel(pdb_data, "pdb")
    view.setStyle({"cartoon": {"color": "spectrum"}})
    view.zoomTo()
    view.render()  # Required for proper rendering before PNG export

    png_data = view.png()
    img = Image.open(BytesIO(png_data))
    img.save(output_path)
    print(f"Saved: {output_path}")
    
# Compare two PDB structures using RMSD
def compute_rmsd(pdb1: dict, pdb2: dict)-> float:
    parser = PDBParser(QUIET=True)
    structure1 = parser.get_structure("structure1", StringIO(pdb1))
    structure2 = parser.get_structure("structure2", StringIO(pdb2))

    # Extract CA atoms for RMSD calculation
    atoms1 = [atom for atom in structure1.get_atoms() if atom.get_id() == "CA"]
    atoms2 = [atom for atom in structure2.get_atoms() if atom.get_id() == "CA"]

    # Match shorter length if needed
    min_len = min(len(atoms1), len(atoms2))
    atoms1 = atoms1[:min_len]
    atoms2 = atoms2[:min_len]

    sup = Superimposer()
    sup.set_atoms(atoms1, atoms2)
    sup.apply(structure2.get_atoms())

    return sup.rms

# Function to compute biophysical properties (e.g., molecular weight, instability index)
def compute_biophysical_properties(sequence: str)-> dict:
    # Placeholder values, ideally we'd use a tool like ExPASy ProtParam for this
    molecular_weight = len(sequence) * 110  # Approx. average amino acid mass (in Da)
    instability_index = len(sequence) % 50  # Placeholder for actual instability index computation
    hydrophobicity = sum(1 for aa in sequence if aa in 'VILFMWY') / len(sequence)  # Hydrophobic index

    return {
        "Molecular Wt. (Da)": molecular_weight,
        "Instability I.": instability_index,
        "Hydrophobicity I.": hydrophobicity
    }

def validate_openai_token(token: str)-> bool:
    """Validate the OpenAI API token by making a test request."""
    test_url = "https://api.openai.com/v1/models"
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(test_url, headers=headers)
    
    if response.status_code == 200:
        st.session_state.valid_token = True
    else:
        st.session_state.valid_token = False
    return response.status_code == 200


def is_new_token(openai_token: str, st: st)-> bool:
    """Check if the provided token is new."""
    if openai_token and openai_token != st.session_state.openai_token:
        st.session_state.openai_token = openai_token
        return True
    return False
    
def is_valid_token(openai_token: str, st: st)-> bool:
    """Check if the provided token is valid."""
    
    if not openai_token:
        return None
    
    if is_new_token(openai_token, st) and validate_openai_token(openai_token):
        st.session_state.valid_token = True
        return True

    return st.session_state.valid_token

def make_filename_safe(text: str) -> str:
    # Lowercase the text
    text = text.lower()
    # Replace spaces and hyphens with underscores
    text = re.sub(r'[\s\-]+', '_', text)
    # Remove any characters that are not alphanumeric or underscores
    text = re.sub(r'[^\w_]', '', text)
    return text

def show_pdb_structure(title: str, sequence: str)-> str:
    """Display the PDB structure using py3Dmol."""
    st.info(title)
    pdb = predict_structure(sequence)
    if pdb:
        show_structure(pdb, title)
        # save_pdb_as_png(pdb, output_path=f"{make_filename_safe(title)}.png")
        # save_pdb_as_png_with_pymol(pdb, output_path=f"{make_filename_safe(title)}_pymol.png")
        plot_pdb_backbone(pdb, output_path=f"/tmp/{make_filename_safe(title)}_backbone_{st.session_state.session_id}.png")
        return pdb
    return None

def get_bio_properties(sequence: str)-> dict:
    # Biophysical properties
    return compute_biophysical_properties(sequence)

def get_bio_prop_evaluation_results(sequence: str)-> dict:
    """Compute and return biophysical properties."""
    bio_properties = get_bio_properties(sequence)
    
    # Evaluate RMSD
    pdb_structures = st.session_state[PDB_STRUCTURES_KEY]
    if len(pdb_structures) > 1:
        rmsd = compute_rmsd(pdb_structures[0], pdb_structures[1])
    else:
        rmsd = None
    
    return {
        "RMSD": rmsd,
        "Molecular Wt. (Da)": bio_properties["Molecular Wt. (Da)"],
        "Instability I.": bio_properties["Instability I."],
        "Hydrophobicity I.": bio_properties["Hydrophobicity I."]
    }

def get_predicted_sequence_with_hypothesis (user_config: dict)-> tuple:
    """Get the predicted sequence with hypothesis."""
    # Assuming you have the MC_NEST_gpt4o class already defined
    mc_nest = MC_NEST_gpt4o(
        background_information=get_background_info(sequence=user_config["sequence_input"].strip()),  # Background information for the sequence
        max_rollouts=user_config["max_rollouts"],  # Number of rollouts to perform
        selection_policy=user_config["selection_policy_value"],  # Selection policy (GREEDY, IMPORTANCE_SAMPLING, etc.)
        initialize_strategy=user_config["zero_shot"],   # Initialization strategy (ZERO_SHOT or DUMMY_ANSWER)
        openai_token=user_config["openai_token"],  # OpenAI API token
    )

    # Run the Monte Carlo NEST algorithm
    best_hypothesis = mc_nest.run()
    
    # Protein sequences (replace these with your actual sequences)
    myc = mc_nest.protein_sequences['modified_sequence']
    
    return best_hypothesis, myc


def get_background_info(sequence: str)-> str:
    segments = {
        "sv40_nls": sequence[0:10],      # MARTKQTARK
        "spacer": sequence[10:35],       # STGGKAPRKQLASKAARKSAARAAAA
        "gly_linker": sequence[35:]      # GGGGGGG
    }
    # Properly formatted background information string
    return f"""
    The sequence {sequence} is a synthetic peptide widely utilized in molecular biology and biomedical research. The first segment, "{segments['sv40_nls']}," is derived from the simian virus 40 (SV40) large T-antigen and functions as a nuclear localization signal (NLS), directing the transport of proteins into the cell nucleus (Kalderon et al., 1984). The following segment, "{segments['spacer']}," acts as a spacer, providing flexibility and minimizing steric hindrance between protein domains when used in fusion proteins, a strategy often employed in protein engineering (Chatterjee et al., 2014). The final part, "{segments['gly_linker']}," consists of six glycine residues and serves as a flexible linker, allowing for free movement between adjacent protein domains (Strop et al., 2008). This peptide is particularly useful for studying nuclear processes, protein-protein interactions, and recombinant protein engineering (Fahmy et al., 2005). It enhances the solubility and functionality of fusion proteins and aids in targeting proteins to the nucleus (Caron et al., 2010). Researchers must consider the potential for non-specific interactions and the context-dependent behavior of synthetic peptides, ensuring the NLS and linker sequences function properly in the specific experimental context (Zhou et al., 2013). Overall, this peptide plays a crucial role in advancing our understanding of protein dynamics and interactions within cellular systems.
    """

def show_organization_logos()-> None:
    """Display logos of supported organizations."""
    st.markdown("---")
    st.markdown("### 🤝 Supported Organizations")
    cols = st.columns(2)

    with cols[0]:
        st.image("./src/images/L3S_Logo.png", width=60)

    with cols[1]:
        st.image("./src/images/Logo_tib.png", width=200)
