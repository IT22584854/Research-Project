import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10

# FIGURE 1.1: Group Project Architecture
def create_figure_1_1():
    fig, ax = plt.subplots(figsize=(12, 14))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 16)
    ax.axis('off')
    
    # Title
    ax.text(5, 15.5, 'Multi-Agent System Architecture', 
            ha='center', fontsize=14, weight='bold')
    
    # Component 1 (YOUR WORK) - Highlighted
    component1 = FancyBboxPatch((0.5, 10.5), 9, 4, 
                                boxstyle="round,pad=0.1", 
                                edgecolor='red', facecolor='#FFE6E6', 
                                linewidth=3)
    ax.add_patch(component1)
    ax.text(5, 14, 'COMPONENT 1: CORPUS ENGINEERING & DATASETS (MY WORK)',
            ha='center', fontsize=11, weight='bold', color='red')
    
    # Corpus box
    corpus_box = FancyBboxPatch((1, 13), 8, 0.6,
                                boxstyle="round,pad=0.05",
                                edgecolor='black', facecolor='lightblue',
                                linewidth=1.5)
    ax.add_patch(corpus_box)
    ax.text(5, 13.3, 'Multilingual Corpus (1,379 docs) | 16 govt websites | Si/Ta/En',
            ha='center', fontsize=9)
    
    # Translation dataset (output of your component, used by others)
    trans_box = FancyBboxPatch((1, 12.2), 8, 0.6,
                               boxstyle="round,pad=0.05",
                               edgecolor='black', facecolor='lightgreen',
                               linewidth=1.5)
    ax.add_patch(trans_box)
    ax.text(5, 12.5, 'Translation Dataset (Si↔En, Ta↔En) → Input to Model Fine-Tuning',
            ha='center', fontsize=9)
    
    # QA dataset (output of your component, used by others)
    qa_box = FancyBboxPatch((1, 11.4), 8, 0.6,
                            boxstyle="round,pad=0.05",
                            edgecolor='black', facecolor='lightyellow',
                            linewidth=1.5)
    ax.add_patch(qa_box)
    ax.text(5, 11.7, 'QA Dataset (5,252 pairs) → Input to Model Fine-Tuning & RAG',
            ha='center', fontsize=9)
    
    # RAG-related note (you do NOT build RAG DB)
    rag_box = FancyBboxPatch((1, 10.6), 8, 0.6,
                             boxstyle="round,pad=0.05",
                             edgecolor='black', facecolor='#FFE6CC',
                             linewidth=1.5)
    ax.add_patch(rag_box)
    ax.text(5, 10.9, 'Clean Corpus + Metadata → Used by Agentic & Eval Components to build RAG DB',
            ha='center', fontsize=9)
    
    # Arrow from your outputs down to Component 2 (they consume your datasets)
    arrow1 = FancyArrowPatch((5, 10.5), (5, 9.8),
                             arrowstyle='->', mutation_scale=20, 
                             linewidth=2, color='black')
    ax.add_patch(arrow1)
    
    # Component 2 – Model Build / Fine-Tuning
    component2 = FancyBboxPatch((0.5, 8.5), 9, 1.2,
                                boxstyle="round,pad=0.1",
                                edgecolor='blue', facecolor='#E6F2FF',
                                linewidth=2)
    ax.add_patch(component2)
    ax.text(5, 9.5, 'COMPONENT 2: MODEL BUILD & FINE-TUNING (Team Member)',
            ha='center', fontsize=11, weight='bold', color='blue')
    ax.text(5, 9, 'Fine-tune base LLM using Translation + QA Datasets from Component 1',
            ha='center', fontsize=9)
    
    # Arrow down to Component 3
    arrow2 = FancyArrowPatch((5, 8.5), (5, 7.8),
                             arrowstyle='->', mutation_scale=20,
                             linewidth=2, color='black')
    ax.add_patch(arrow2)
    
    # Component 3 – Agentic Framework (they build/own RAG DB)
    component3 = FancyBboxPatch((0.5, 6), 9, 1.7,
                                boxstyle="round,pad=0.1",
                                edgecolor='green', facecolor='#E6FFE6',
                                linewidth=2)
    ax.add_patch(component3)
    ax.text(5, 7.5, 'COMPONENT 3: AGENTIC FRAMEWORK (Team Member)',
            ha='center', fontsize=11, weight='bold', color='green')
    ax.text(5, 7.05, 'Query Understanding | RAG Retrieval | Answer Generation',
            ha='center', fontsize=9)
    ax.text(5, 6.6, 'Uses: Fine-tuned Model + RAG Database built from Corpus/QA',
            ha='center', fontsize=9, style='italic')
    
    # Arrow down to Component 4
    arrow3 = FancyArrowPatch((5, 6), (5, 5.3),
                             arrowstyle='->', mutation_scale=20,
                             linewidth=2, color='black')
    ax.add_patch(arrow3)
    
    # Component 4 – Evaluation & Benchmarking (they also use corpus/QA)
    component4 = FancyBboxPatch((0.5, 3.8), 9, 1.4,
                                boxstyle="round,pad=0.1",
                                edgecolor='purple', facecolor='#F2E6FF',
                                linewidth=2)
    ax.add_patch(component4)
    ax.text(5, 5, 'COMPONENT 4: EVALUATION & BENCHMARKING (Team Member)',
            ha='center', fontsize=11, weight='bold', color='purple')
    ax.text(5, 4.5, 'Test Fine-Tuned Model + RAG DB using QA / Corpus-Based Benchmarks',
            ha='center', fontsize=9)
    
    # User query at top
    user_box = FancyBboxPatch((2, 15.8), 6, 0.5,
                              boxstyle="round,pad=0.1",
                              edgecolor='black', facecolor='yellow',
                              linewidth=2)
    ax.add_patch(user_box)
    ax.text(5, 16.05, 'USER QUERY (Sinhala / Tamil / English)',
            ha='center', fontsize=10, weight='bold')
    
    # Answer at bottom
    answer_box = FancyBboxPatch((2, 3), 6, 0.5,
                                boxstyle="round,pad=0.1",
                                edgecolor='black', facecolor='lightgreen',
                                linewidth=2)
    ax.add_patch(answer_box)
    ax.text(5, 3.25, 'ANSWER TO USER (Original Language)',
            ha='center', fontsize=10, weight='bold')
    
    plt.tight_layout()
    plt.savefig(r"C:\Users\Charunya\Downloads\Figure_1_1_Architecture.png",
            dpi=300, bbox_inches='tight')
    print("✓ Figure 1.1 saved as 'Figure_1_1_Architecture.png'")
    plt.close()