import sys
import os

# Ensure we can import from agentic_workflow
current_dir = os.path.dirname(os.path.abspath(__file__))
workflow_dir = os.path.join(current_dir, "agentic_workflow")
sys.path.append(workflow_dir)

from agent import MyAgent


def generate_graph():
    print("Initializing Agent...")
    agent = MyAgent()
    print("Compiling Workflow...")
    compiled_graph = agent.workflow.compile()

    print("Generating Mermaid PNG...")
    try:
        png_data = compiled_graph.get_graph().draw_mermaid_png()

        # Output to ../docs/graph_visualization.png
        output_dir = os.path.join(os.path.dirname(current_dir), "docs")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "graph_visualization.png")

        with open(output_path, "wb") as f:
            f.write(png_data)

        print(f"✅ Graph saved to {output_path}")

    except Exception as e:
        print(f"❌ Failed to generate PNG: {e}")
        # Fallback to printing Mermaid syntax
        print("Mermaid Syntax:")
        print(compiled_graph.get_graph().draw_mermaid())


if __name__ == "__main__":
    generate_graph()
