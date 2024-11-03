import streamlit as st
from barfi import st_barfi, Block
import yaml
import json
import os
import random

# Set page config at the very beginning
st.set_page_config(layout="wide")

def generate_random_color():
    """Generate a random pastel color"""
    hue = random.random()  # Random hue
    saturation = 0.3 + random.random() * 0.2  # 30-50% saturation
    value = 0.9 + random.random() * 0.1  # 90-100% value
    
    # Convert HSV to RGB
    import colorsys
    rgb = colorsys.hsv_to_rgb(hue, saturation, value)
    
    # Convert to hex
    return '#{:02x}{:02x}{:02x}'.format(
        int(rgb[0] * 255),
        int(rgb[1] * 255),
        int(rgb[2] * 255)
    )

class DynamicBlock(Block):
    def __init__(self, name, config):
        super().__init__(name)
        
        # Add inputs
        for input_name, input_type in config.get('input', {}).items():
            self.add_input(input_name)
            
        # Add outputs
        for output_name, output_type in config.get('output', {}).items():
            self.add_output(output_name)
            
        # Set random color based on block type
        if name.lower() == 'query input':
            self.color = '#90EE90'  # Light green for input
        elif name.lower() == 'response output':
            self.color = '#FFA500'  # Orange for output
        else:
            self.color = generate_random_color()

def create_pipeline_webapp():
    st.title("RAG Pipeline Builder")

    # Load component configurations
    with open("components.yaml", 'r') as f:
        components_config = yaml.safe_load(f)['components']

    # Create blocks dynamically from config
    base_blocks = []
    
    # Create a color map for consistent colors across refreshes
    if 'color_map' not in st.session_state:
        st.session_state.color_map = {
            'query_input': '#90EE90',
            'response_output': '#FFA500'
        }

    # Process regular components
    for component_name, component_config in components_config.items():
        if component_name != 'retrievers':  # Skip retrievers as they're handled separately
            block = DynamicBlock(
                name=component_name.replace('_', ' ').title(),
                config=component_config
            )
            block.color = st.session_state.color_map.get(
                component_name, 
                generate_random_color()
            )
            base_blocks.append(block)

    # Process retrievers
    if 'retrievers' in components_config:
        for retriever in components_config['retrievers']:
            retriever_type = retriever['type']
            block_name = f"{retriever_type.title()} Retriever"
            
            if block_name not in st.session_state.color_map:
                st.session_state.color_map[block_name] = generate_random_color()
                
            block = DynamicBlock(
                name=block_name,
                config=retriever
            )
            block.color = st.session_state.color_map[block_name]
            base_blocks.append(block)

    # Create the barfi instance with block instances
    flow = st_barfi(base_blocks=base_blocks)

    # Add save functionality in a sidebar
    with st.sidebar:
        if st.button("Save Pipeline"):
            try:
                if flow and isinstance(flow, dict):
                    nodes = {}
                    edges = []
                    
                    # Process nodes
                    for node_id, node_data in flow.items():
                        component_type = node_data['type'].lower().replace(' ', '_')
                        
                        # Handle retriever parameters
                        if 'retriever' in component_type:
                            retriever_type = component_type.split('_')[0]  # e.g., 'vector' from 'vector_retriever'
                            parameters = next(
                                (r['parameters'] for r in components_config['retrievers'] 
                                 if r['type'] == retriever_type),
                                {}
                            )
                        else:
                            parameters = components_config.get(component_type, {}).get('parameters', {})

                        nodes[node_id] = {
                            'type': component_type,
                            'parameters': parameters,
                            'interfaces': {}
                        }
                        
                        # Process interfaces
                        if 'interfaces' in node_data:
                            for interface_name, interface_data in node_data['interfaces'].items():
                                interface_type = interface_data['type']
                                nodes[node_id]['interfaces'][interface_name] = interface_type
                                
                                if interface_type == 'output' and 'to' in interface_data:
                                    for target_id, target_port in interface_data['to'].items():
                                        edges.append({
                                            'source': node_id,
                                            'source_port': interface_name,
                                            'target': target_id,
                                            'target_port': target_port
                                        })
                    
                    pipeline_schema = {
                        'nodes': nodes,
                        'edges': edges,
                        'config': components_config
                    }
                    
                    save_path = os.path.abspath('schema.barfi')
                    with open(save_path, 'w', encoding='utf-8') as f:
                        json.dump(pipeline_schema, f, ensure_ascii=False, indent=2)
                    
                    st.success(f"Pipeline saved successfully to: {save_path}")
                    
                    with st.expander("Saved Pipeline Structure"):
                        st.json(pipeline_schema)
                else:
                    st.warning("No pipeline data to save. Please create a pipeline first.")
                    
            except Exception as e:
                st.error(f"Error saving pipeline: {str(e)}")
                st.write("Error details:", e.__class__.__name__)

if __name__ == "__main__":
    create_pipeline_webapp()