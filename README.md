
# ScHiCAtt: Enhancing Single-Cell Hi-C Data Resolution Using Attention-Based Models

___________________  
#### OluwadareLab, University of Colorado, Colorado Springs  
___________________

#### Developers:  
Rohit Menon  
Department of Computer Science  
University of Colorado, Colorado Springs
Email: rmenonch@uccs.edu

#### Contact:  
Oluwatosin Oluwadare, PhD
Department of Computer Science 
University of Colorado, Colorado Springs
Email: ooluwada@uccs.edu
___________________  

## Overview:
**ScHiCAtt** is a deep learning model designed to enhance the resolution of Single-Cell Hi-C contact matrices using various attention mechanisms, such as self-attention, local attention, global attention, and dynamic attention. The model leverages GAN-based training to optimize the quality of Hi-C contact maps through a composite loss function consisting of MSE, perceptual, total variation, and adversarial losses.

___________________  

## Build Instructions:

ScHiCAtt runs in a Docker-containerized environment. Follow these steps to set up ScHiCAtt.

1. Clone this repository:

```bash
git clone https://github.com/OluwadareLab/ScHiCAtt.git && cd ScHiCAtt
```

2. Pull the ScHiCAtt Docker image:

```bash
docker pull oluwadarelab/schicatt:latest
```

3. Run the container and mount the present working directory to the container:

```bash
docker run --rm --gpus all -it --name schicatt -v ${PWD}:${PWD} oluwadarelab/schicatt
```

4. You can now navigate within the container and run the model.

___________________  

## Dependencies:

All necessary dependencies are bundled within the Docker environment. The core dependencies include:

- Python 3.8
- PyTorch 1.10.0 (CUDA 11.3)
- NumPy 1.21.1
- SciPy 1.7.0
- Pandas 1.3.1
- Scikit-learn 0.24.2
- Matplotlib 3.4.2
- tqdm 4.61.2

**_Note:_** GPU usage for training and testing is highly recommended.


___________________  

##  Training

To train the ScHiCAtt model on your Hi-C data, navigate to the `Training folder` and use the following command:

```
python3 train.py \
  --train_data path/to/train_data.npz \
  --valid_data path/to/valid_data.npz \
  --epochs 50 \
  --batch_size 32 \
  --lr 0.0001 \
  --save_path checkpoints/schicatt.pth
```

###  Input Format

Your `.npz` files must contain the following keys:
- `'data'`: Low-resolution Hi-C patches (shape: `[N, 1, 40, 40]`)
- `'target'`: Ground truth high-resolution patches (same shape)

###  Arguments

| Argument        | Description                                      | Default        |
|----------------|--------------------------------------------------|----------------|
| `--train_data`  | Path to the training `.npz` dataset              | **Required**   |
| `--valid_data`  | Path to the validation `.npz` dataset            | **Required**   |
| `--epochs`      | Number of training epochs                        | `1`            |
| `--batch_size`  | Batch size for training                          | `64`           |
| `--lr`          | Learning rate for the optimizer                  | `0.0003`       |
| `--save_path`   | Path to save the best model checkpoint           | `schicatt.pth` |

###  Output

The script saves the best-performing model (based on validation loss) to the specified path.

By default, the model will be saved to `schicatt.pth`.

___________________  

## Inference

After training the model, you can run inference (present in the `Training` folder) using the saved checkpoint on test data using:

```
python3 infer.py \
  --input path/to/test_data.npz \
  --checkpoint path/to/schicatt.pth \
  --output path/to/output_directory \
  --cuda 0
```

### Input Format

The input `.npz` file must contain the following keys:
- `'data'`: Low-resolution Hi-C patches (shape: `[N, 1, 40, 40]`)
- `'inds'`: Index array indicating patch positions and chromosome IDs (shape: `[N, 4]`)

### Arguments

| Argument         | Description                                                  | Default     |
|------------------|--------------------------------------------------------------|-------------|
| `--input`        | Path to input `.npz` test dataset                            | **Required**|
| `--checkpoint`   | Path to the trained model checkpoint `.pth` file             | **Required**|
| `--output`       | Directory to save reconstructed full matrices                | **Required**|
| `--multi-chrom`  | (Optional) Enable multi-chromosome handling if needed       | `False`     |
| `--cuda`         | CUDA device ID (e.g., `0` for GPU, `-1` for CPU)             | `-1`        |

### Output

The script reconstructs chromosome-wise Hi-C matrices and saves them as compressed `.npz` files inside the specified output directory.

Each output file is named as:
```
chr<chromosome_id>_schicatt.npz
```
and contains the key `'schicatt'` with the predicted high-resolution matrix.

### Example

```
python3 infer_schicatt.py \
  --input data/test_chr11.npz \
  --checkpoint schicatt.pth \
  --output results/ \
  --cuda 0
```

This will generate the file `results/chr11_schicatt.npz` containing the inferred matrix.


___________________  

## Analysis

All the analysis scripts are available at **analysis** folder
### TAD Plots
#### TAD detection with deDoc2
* We used `https://github.com/zengguangjie/deDoc2` for TAD detection from scHiC data. We used lower TAD-like domains.
* Download the **doDoc2**
* Edit the necessary variables in `call_tads.py` script such as *INPUT_FILEPATH* and adjust other things as necessary.
* Run `python3 call_tads.py`

#### TAD plot
* We used `https://xiaotaowang.github.io/TADLib/index.html` to produce the TAD figures.
* Here, we provided a sample python script `draw_tad_plots.py` to produce the plots. Update *INPUT_FILEPATH*, *MATRIX_FILEPATH*, *FILENAMES*, *CHROMOSOMES*, *ALGORITHMS*, *OUTPUT_PATH*.
* It takes matrix and TADs as input. TAD file structure should be (without heading):

| Chromosome | Start Position | End Position |
|------------|----------------|--------------|
| chr12      | 80000         | 1960000    |
| chr12      | 2000000      | 2720000    |
| chr12      | 2760000      | 4040000    |
| chr12      | 4080000      | 5480000    |
| chr12      | 5520000      | 5720000    |

* Run `python3 draw_tad_plots.py`
   
### L2 norm
* Edit draw_l2_norm.py with your filenames and paths including *MATRIX_FILEPATH*, *FILENAMES*, *CHROMOSOMES*, *ALGORITHMS*.
* It takes matrix as input.
* Run `python3 draw_l2_norm.py`

___________________  

## Cite:

If you use ScHiCAtt in your research, please cite the following:

Rohit Menon, Oluwatosin Oluwadare, *ScHiCAtt: Enhancing Single-Cell Hi-C Data with Attention Mechanisms*, [Journal Name], [Year], [DOI link if available].

___________________  

## License:

This project is licensed under the MIT License. See the `LICENSE` file for details.
