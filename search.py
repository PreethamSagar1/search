import streamlit as st
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import easyocr
import tempfile
import os


class ProfileManager:
    """Handles profile-related operations such as adding, updating, and deleting profiles."""
    def _init_(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')  # For embeddings
        self.reader = easyocr.Reader(['en'])  # For text extraction
        self.profiles_db = pd.DataFrame(columns=["id", "name", "education", "resume_text", "embedding"])
        self.index = faiss.IndexFlatL2(384)  # FAISS index for 384-dimensional embeddings
        self.profile_id_map = {}  # Mapping from FAISS indices to profile IDs
        self.current_id = 1  # To generate unique profile IDs

    def process_resume(self, resume_path):
        """Extract text from a resume using EasyOCR and generate an embedding."""
        with open(resume_path, "rb") as f:
            file_bytes = f.read()
        # OCR to extract text
        text = " ".join(self.reader.readtext(file_bytes, detail=0))
        # Generate embedding
        embedding = self.model.encode(text).astype('float32')
        return text, embedding

    def add_profile(self, name, education, resume_path):
        """Add a new profile to the database and FAISS index."""
        try:
            # Extract resume text and embedding
            resume_text, embedding = self.process_resume(resume_path)
            # Add profile to the database
            profile_id = self.current_id
            self.profiles_db = pd.concat([
                self.profiles_db,
                pd.DataFrame({
                    "id": [profile_id],
                    "name": [name],
                    "education": [education],
                    "resume_text": [resume_text],
                    "embedding": [embedding]
                })
            ], ignore_index=True)
            # Add embedding to FAISS index
            self.index.add(np.array([embedding]))
            self.profile_id_map[self.index.ntotal - 1] = profile_id
            self.current_id += 1
            return f"Profile '{name}' added successfully!"
        except Exception as e:
            return f"Error while adding profile: {str(e)}"

    def search_profiles(self, query, top_k):
        """Search profiles based on a query embedding."""
        query_embedding = self.model.encode(query).astype('float32')
        distances, indices = self.index.search(np.array([query_embedding]), top_k)
        results = []
        for idx in indices[0]:
            profile_id = self.profile_id_map.get(idx)
            if profile_id is not None:
                profile = self.profiles_db[self.profiles_db['id'] == profile_id].iloc[0]
                results.append({
                    "id": profile.id,
                    "name": profile.name,
                    "education": profile.education,
                    "resume_text": profile.resume_text
                })
        return results

    def update_profile(self, profile_id, name=None, education=None, resume_path=None):
        """Update a profile and its embedding."""
        try:
            profile_index = self.profiles_db[self.profiles_db['id'] == profile_id].index[0]
            if name:
                self.profiles_db.at[profile_index, 'name'] = name
            if education:
                self.profiles_db.at[profile_index, 'education'] = education
            if resume_path:
                resume_text, embedding = self.process_resume(resume_path)
                self.profiles_db.at[profile_index, 'resume_text'] = resume_text
                self.profiles_db.at[profile_index, 'embedding'] = embedding
                # Update FAISS
                faiss_idx = [idx for idx, pid in self.profile_id_map.items() if pid == profile_id][0]
                self.index.reconstruct(faiss_idx)
                self.index.replace_ids([faiss_idx], np.array([embedding]))
            return f"Profile '{profile_id}' updated successfully!"
        except Exception as e:
            return f"Error while updating profile: {str(e)}"

    def delete_profile(self, profile_id):
        """Delete a profile and remove its embedding from FAISS."""
        try:
            profile_index = self.profiles_db[self.profiles_db['id'] == profile_id].index[0]
            # Remove from database
            self.profiles_db = self.profiles_db.drop(profile_index).reset_index(drop=True)
            # Remove from FAISS
            faiss_idx = [idx for idx, pid in self.profile_id_map.items() if pid == profile_id][0]
            self.index.remove_ids(np.array([faiss_idx]))
            del self.profile_id_map[faiss_idx]
            return f"Profile '{profile_id}' deleted successfully!"
        except Exception as e:
            return f"Error while deleting profile: {str(e)}"


class AppFeatures:
    """Manages the Streamlit app workflow."""
    def _init_(self):
        self.manager = ProfileManager()

    def add_profile_ui(self):
        """UI for adding profiles."""
        st.header("Add New Profile")
        name = st.text_input("Name")
        education = st.text_area("Education")
        resume_file = st.file_uploader("Upload Resume", type=["pdf"])
        add_button = st.button("Add Profile")

        if add_button:
            if not name or not education or not resume_file:
                st.error("Please fill in all fields.")
            else:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                    temp_file.write(resume_file.read())
                    result = self.manager.add_profile(name, education, temp_file.name)
                os.remove(temp_file.name)
                st.success(result)

    def search_profiles_ui(self):
        """UI for searching profiles."""
        st.header("Search Profiles")
        query = st.text_input("Enter your search query")
        top_k = st.slider("Number of results", 1, 50, 10)
        search_button = st.button("Search")

        if search_button:
            if not query:
                st.error("Please enter a search query.")
            else:
                results = self.manager.search_profiles(query, top_k)
                if results:
                    for profile in results:
                        st.subheader(f"Profile: {profile['name']}")
                        st.write(f"Education: {profile['education']}")
                        st.write(f"Resume: {profile['resume_text']}")
                else:
                    st.warning("No profiles found.")

    def update_delete_profiles_ui(self):
        """UI for updating and deleting profiles."""
        st.header("Update or Delete Profiles")
        profile_id = st.number_input("Enter Profile ID", min_value=1, step=1)
        action = st.radio("Select an Action", ["Update", "Delete"])

        if action == "Update":
            name = st.text_input("New Name (leave blank to keep unchanged)")
            education = st.text_area("New Education (leave blank to keep unchanged)")
            resume_file = st.file_uploader("Upload New Resume (optional)", type=["pdf"])
            update_button = st.button("Update Profile")

            if update_button:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                    temp_file.write(resume_file.read() if resume_file else b"")
                    result = self.manager.update_profile(profile_id, name, education, temp_file.name if resume_file else None)
                os.remove(temp_file.name)
                st.success(result)

        elif action == "Delete":
            delete_button = st.button("Delete Profile")
            if delete_button:
                result = self.manager.delete_profile(profile_id)
                st.success(result)


def main():
    """Main function to run the app."""
    st.title("Profile Management and Search System")
    features = AppFeatures()

    action = st.sidebar.radio("Select an Action", ["Add Profile", "Search Profiles", "Update/Delete Profile"])

    if action == "Add Profile":
        features.add_profile_ui()
    elif action == "Search Profiles":
        features.search_profiles_ui()
    elif action == "Update/Delete Profile":
        features.update_delete_profiles_ui()


if _name_ == "_main_":
    main()