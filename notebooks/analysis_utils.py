import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

def get_category_images(category_path):
    """
    Returns a list of valid image filenames found in the given directory.
    Supported formats: .jpg, .jpeg, .png, .bmp, .gif.
    """
    if not category_path.exists():
        return []
    
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
    return [f.name for f in category_path.iterdir() 
            if f.is_file() and f.suffix.lower() in valid_extensions]

def get_category_names(data_dir):
    """
    Retrieves animal category names from the index file or the directory structure.
    """
    path_names_category = data_dir / 'name of the animals.txt'
    
    # Try to load names from the text file first
    if path_names_category.exists():
        with open(path_names_category, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    
    # Fallback: Use folder names if the text file is missing
    animals_dir = data_dir / 'mini_animals' / 'animals'
    if animals_dir.exists():
        return sorted([d.name for d in animals_dir.iterdir() if d.is_dir()])
    
    return []

def calculate_rgb_averages(data_dir):
    """
    Calculates the mean R, G, and B values for every image in each category.
    """
    category_names = get_category_names(data_dir)
    animals_dir = data_dir / 'mini_animals' / 'animals'
    rgb_means = {}

    for animal in category_names:
        animal_path = animals_dir / animal
        images = get_category_images(animal_path)
        
        if not images:
            continue
            
        class_means = []
        for img_name in images:
            img_path = animal_path / img_name
            try:
                with Image.open(img_path) as img:
                    img = img.convert("RGB")
                    arr = np.array(img, dtype=np.float32)
                    # Compute mean across spatial dimensions (Height and Width)
                    img_mean = arr.mean(axis=(0, 1))
                    class_means.append(img_mean)
            except Exception as e:
                print(f"Skipping {img_path.name}: {e}")
        
        if class_means:
            rgb_means[animal] = np.mean(class_means, axis=0)
            
    return rgb_means

def get_df_with_rgb_avg(rgb_means):
    """
    Processes RGB data into a structured DataFrame with luminance and colorfulness metrics.
    """
    if not rgb_means:
        return pd.DataFrame()

    df = pd.DataFrame.from_dict(rgb_means, orient='index', columns=['R', 'G', 'B'])
    df.index.name = 'class'
    df = df.reset_index()
    
    # Normalize RGB values to [0, 1] range
    df[['R', 'G', 'B']] /= 255.0
    
    # Calculate Perceptual Luminance (ITU-R BT.709)
    df['luminance'] = 0.2126 * df['R'] + 0.7152 * df['G'] + 0.0722 * df['B']
    
    # Approximate colorfulness using R-G and Y-B components
    rg = df['R'] - df['G']
    yb = 0.5 * (df['R'] + df['G']) - df['B']
    df['colorfulness'] = np.sqrt(rg**2 + yb**2) + 0.3 * np.sqrt(rg.var() + yb.var())
    
    # Save processed metrics for later use
    # Note: Saved to current working directory
    df.to_csv('class_rgb_averages.csv', index=False)
    return df

def plot_rgb_heatmap(df, output_path):
    """
    Plots a heatmap showing R, G, B distributions per class, sorted by luminance.
    """
    if df.empty: return
    
    df_sorted = df.sort_values('luminance').set_index('class')
    
    plt.figure(figsize=(8, 14))
    sns.heatmap(df_sorted[['R', 'G', 'B']], annot=False, cmap='viridis')
    plt.title('Color Distribution per Class (Ordered by Luminance)')
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path / "heatmap_by_class.png")
    plt.show()

def plot_clustermap(df, output_path):
    """
    Performs hierarchical clustering on classes based on their color profiles.
    """
    if df.empty: return
    
    plot_df = df.set_index('class')[['R', 'G', 'B']]
    g = sns.clustermap(
        plot_df, 
        method='ward', 
        metric='euclidean', 
        cmap='viridis', 
        figsize=(10, 15)
    )
    plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=8)
    plt.suptitle('Hierarchical Clustering by Color Profile', y=1.02)
    if output_path:
        plt.savefig(output_path / "clustermap_rgb.png")
    plt.show()

def plot_avg_colors(df, output_path):
    """
    Generates a grid displaying the actual average color calculated for each class.
    """
    if df.empty: return
    
    n_classes = len(df)
    cols = 6
    rows = int(np.ceil(n_classes / cols))
    
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 2 * rows))
    axes = axes.flatten()
    
    for i, (idx, row) in enumerate(df.iterrows()):
        ax = axes[i]
        ax.axis('off')
        ax.set_title(row['class'], fontsize=8)
        color = np.clip([row['R'], row['G'], row['B']], 0, 1)
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, color=color))
        
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
        
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path / "color_averages.png")
    plt.show()

if __name__ == '__main__':
    # Determine directory structure relative to this script
    current_dir = Path(__file__).parent.resolve()
    project_root = current_dir.parent
    
    # Define and create output directories
    plots_dir = project_root / 'reports' / 'figures'
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    data_dir = project_root / 'data'
    
    print("🎨 Starting exploratory color analysis from script...")
    
    averages = calculate_rgb_averages(data_dir)
    df_metrics = get_df_with_rgb_avg(averages)
    
    if not df_metrics.empty:
        print(f"📊 Generating visualizations in {plots_dir}...")
        plot_rgb_heatmap(df_metrics, plots_dir)
        plot_clustermap(df_metrics, plots_dir)
        plot_avg_colors(df_metrics, plots_dir)
        print("✅ Analysis complete.")
    else:
        print("❌ No valid data found. Please ensure the dataset exists in 'data/mini_animals'.")