import seaborn as sns
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pdoc
import os
#import scipy
#import torch


def get_category_images(category_path):
    """
    Obtiene la lista de imágenes de una categoría específica.

    Parameters
    ----------
    category_path : str or Path
        Ruta a la carpeta de la categoría.

    Returns
    -------
    list of str
        Lista con los nombres de archivos de imágenes válidas 
        (.jpg, .jpeg, .png, .bmp, .gif) en la carpeta.
    """
    imagenes = [f for f in os.listdir(category_path)
                if os.path.isfile(os.path.join(category_path, f))
                and f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))]
    return imagenes

def get_category_names(data_dir):
    """
    Lee los nombres de las categorías desde un archivo de texto.

    Parameters
    ----------
    data_dir : Path
        Ruta al directorio que contiene el archivo 'name of the animals.txt'.

    Returns
    -------
    list of str
        Lista con los nombres de las categorías (una por línea en el archivo).
    """
    path_names_category = data_dir / 'name of the animals.txt'
    with open(path_names_category, 'r') as f:
        category_names = [line.strip() for line in f if line.strip()]
    return category_names

def calculate_rgb_averages(data_dir):
    """
    Calcula los valores promedio de R, G y B por categoría de imágenes.

    Parameters
    ----------
    data_dir : Path
        Ruta al directorio de datos que contiene las imágenes dentro de 'mini_animals'.

    Returns
    -------
    dict
        Diccionario con las categorías como claves y los promedios [R, G, B] como valores.
    """
    category_names = get_category_names(data_dir)
    animals_dir = data_dir / 'mini_animals'
    # Diccionario para guardar la media por clase
    medias_rgb = {}
    for animal in category_names:
        imagenes = get_category_images(animals_dir / animal)
        # Listas para acumular valores por canal
        canales = []
        #Accedemos a la carpeta de la categoria
        carpeta_categoria = os.path.join(animals_dir, animal)
    
        for img_path in imagenes:
            ruta_foto = os.path.join(carpeta_categoria, img_path)
    
            img = Image.open(ruta_foto).convert("RGB")   # asegurar RGB
            arr = np.array(img, dtype=np.float32)
    
            # Calcular media por canal
            r_mean = arr[:, :, 0].mean()
            g_mean = arr[:, :, 1].mean()
            b_mean = arr[:, :, 2].mean()
            canales.append([r_mean, g_mean, b_mean])
    
        # Promedio total de la clase
        if canales:
            medias_rgb[animal] = np.mean(canales, axis=0)
    return medias_rgb

def plot_rgb_heatmap(df, path):
    """
    Genera un mapa de calor (heatmap) de valores promedio R, G, B por clase.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame con columnas ['R', 'G', 'B', 'luminance'].
    path : Path
        Ruta donde se guardará la figura.
    """
    # Ordenar df_norm completo por luminancia
    df_sorted = df.sort_values('luminance')
    
    # Ahora pasamos solo R, G, B al heatmap
    plt.figure(figsize=(6, 12))
    sns.heatmap(df_sorted[['R','G','B']], annot=False, cmap='viridis')
    plt.title('Heatmap: R/G/B por clase (ordenadas por luminancia)')
    plt.xlabel('Canal')
    plt.ylabel('Clase (ordenada)')
    plt.tight_layout()
    plt.savefig(path / "heatmap_by_class.png")
    plt.show()

def plot_clustermap(df, path):
    """
    Genera un clustermap jerárquico de los valores R, G y B.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame con columnas ['R', 'G', 'B'] normalizadas.
    path : Path
        Ruta donde se guardará la figura.
    """
    sns.clustermap(
        df_norm[['R','G','B']], 
        method='ward', metric='euclidean', 
        cmap='viridis', figsize=(8,12)
    )
    plt.suptitle('Clustermap de R/G/B (agrupamiento jerárquico)')
    plt.savefig(path / "clustermap_rgb.png")
    plt.show()

def plot_avg_colors(df, path):
    """
    Genera una visualización de los colores promedio por clase.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame con columnas ['class', 'R', 'G', 'B'].
    path : Path
        Ruta donde se guardará la figura.
    """
    fig, axes = plt.subplots(nrows=15, ncols=6, figsize=(10, 20))  # ajustar rows/cols
    axes = axes.flatten()
    for ax_i, (cls, row) in zip(axes, df.iterrows()):
        ax_i.axis('off')
        ax_i.set_title(cls, fontsize=6)
        ax_i.add_patch(plt.Rectangle((0,0),1,1, color=row[['R','G','B']]/255.0))
    plt.tight_layout()
    plt.savefig(path / "color_averages.png")
    plt.show()

def get_df_with_rgb_avg(medias_rgb):
    """
    Convierte un diccionario de promedios RGB en un DataFrame y añade métricas adicionales.

    Parameters
    ----------
    medias_rgb : dict
        Diccionario con categorías como claves y valores [R, G, B] promedios.

    Returns
    -------
    pandas.DataFrame
        DataFrame con columnas:
        - 'class': nombre de la clase.
        - 'R', 'G', 'B': valores normalizados (0-1).
        - 'luminance': luminancia perceptual.
        - 'colorfulness': medida de colorfulness aproximada.
    """
    df = pd.DataFrame.from_dict(medias_rgb, orient='index', columns=['R','G','B'])
    df.index.name = 'class'
    
    # Pasar 'class' del índice a columna
    df = df.reset_index()
    
    print(df.head())
    print(df.columns)
    df_norm = df.copy()
    df_norm[['R','G','B']] = df[['R','G','B']] / 255.0
    
    # Calcular luminancia aproximada (perceptual)
    df_norm['luminance'] = 0.2126*df_norm['R'] + 0.7152*df_norm['G'] + 0.0722*df_norm['B']
    # Colorfulness (Hasler & Süsstrunk) — aproximación con R-G y Y-B
    rg = df_norm['R'] - df_norm['G']
    yb = 0.5*(df_norm['R'] + df_norm['G']) - df_norm['B']
    df_norm['colorfulness'] = np.sqrt(rg**2 + yb**2) + 0.3 * np.sqrt(rg.var() + yb.var())
    # Guardar para uso futuro
    df_norm.to_csv('medias_rgb_por_clase.csv')
    return df_norm


from pathlib import Path

if __name__ == '__main__':
    project_dir = Path.cwd().parent
    plots_dir = project_dir / 'reports' / 'figures'
    data_dir = project_dir / 'data'
    rgb_averages = calculate_rgb_averages(data_dir)
    df_norm = get_df_with_rgb_avg(rgb_averages)
    plot_rgb_heatmap(df_norm, plots_dir)
    plot_clustermap(df_norm, plots_dir)
    plot_avg_colors(df_norm, plots_dir)