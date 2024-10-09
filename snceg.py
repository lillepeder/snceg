import argparse
from pathlib import Path
from fastMONAI.vision_inference import _do_resize, _to_original_orientation
from fastMONAI.vision_all import load_learner, load_variables, med_img_reader, do_pad_or_crop
from huggingface_hub import snapshot_download



def _inference(learn_inf, reorder, resample, fn,
              save_path=None): 
    """
    Bug-fixed and simplified version of fastMONAI.vision_inference.inference.
    Predict on new data using exported model.
    """             
    org_img, input_img, org_size = med_img_reader(fn, reorder=reorder, resample=resample, 
                                                      only_tensor=False)
    
    pred, *_ = learn_inf.predict(input_img.data)
    
    pred_mask = do_pad_or_crop(pred.float(), input_img.shape[1:], padding_mode=0, 
                               mask_name=None)
    input_img.set_data(pred_mask)
    
    input_img = _do_resize(input_img, org_size, image_interpolation='nearest')
    
    reoriented_array = _to_original_orientation(input_img.as_sitk(), 
                                                ('').join(org_img.orientation))
    
    org_img.set_data(reoriented_array)

    if save_path:
        org_img.save(save_path)
    
    return org_img


def get_model(model_dir):
    model_fn  = 'SNceg-0.1.pkl'
    var_pkl = 'vars_' + model_fn

    if not (model_dir / model_fn).exists():
        print(f'Could not find model {model_dir / model_fn}. Downloading into {model_dir}.')
        snapshot_download(repo_id="lillepeder/SNceg-0.1", local_dir=model_dir)
    
    learner = load_learner(model_dir / model_fn)
    size, reorder, resample = load_variables(pkl_fn=model_dir / var_pkl)
    return learner, size, reorder, resample



def run_one_sample(fn, learner, reorder=None, resample=None, outdir=None, pred_fn=None):
    """
    Wrapper around _inference. Run the model on one image. 
    fn : filename of input image
    pred_fn : desired path of output prediction
    """
    if isinstance(fn, str):
        fn = Path(fn)
    
    if outdir is None:
        outdir = fn.parent

    if pred_fn is None:    
        pred_fn = outdir / f"SN_pred_{fn.name}"

    if Path(pred_fn).exists():
        print('File', pred_fn, 'already exists. Skipping.')
        return pred_fn

    # RUN PREDICTION
    pred = _inference(learn_inf=learner,
                      reorder=reorder,
                      resample=resample, 
                      fn=fn, 
                      save_path=pred_fn)

    print("Saved output file: ", pred_fn)
    
    return pred_fn


def run(inp, output, target_dir, do_resample):
    
    # File handling
    inp = Path(inp).absolute().resolve()

    # if output arg was entered
    if output:
        output = Path(output).absolute().resolve()
    # if target_dir arg was entered
    elif target_dir: 
        target_dir = Path(target_dir).absolute().resolve()
        if not target_dir.exists():
            target_dir.mkdir(parents=True, exist_ok=True)

    
    # if directory was entered for input arg 
    if inp.is_dir():
        input_files = inp.glob('*.nii.gz')
    else:
        input_files = [inp]

    # Load the model
    model_dir = Path('./models').absolute().resolve()
    
    if do_resample == True:
        learner, size, reorder, resample = get_model(model_dir)
    else: 
        learner, size, reorder, _ = get_model(model_dir)
        resample = None

    # Run the model
    for fn in input_files:
        # run silent
        print(fn, target_dir, output)
        run_one_sample(fn, learner, reorder=reorder, resample=resample, outdir=target_dir, pred_fn=output)


def main():
    parser = argparse.ArgumentParser(description='Run SN segmentation algorithm.')
    
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='The filename of the input image, or a directory containing multiple images. In the latter case, you must provide "--target_dir" and leave "--output" empty.')
    parser.add_argument('-o', '--output', type=str, required=False,
                        help='The desired filename of the output image.')
    parser.add_argument('-t', '--target_dir', type=str, required=False,
                        help='Target directory of the output image(s). If entered, "--output" should be left empty.')
    parser.add_argument('-r', '--resample', action="store_true", 
                        help='Whether to resample the image prior to prediction. Should only be left out under special circumstances.')
    
    args = parser.parse_args()
    
    # Ensure arguments were entered correctly
    assert not (bool(args.output) & bool(args.target_dir)), "You entered both --output and --target_dir. You can only enter either one or neither."

    run(args.input, args.output, args.target_dir, args.resample)

if __name__ == "__main__":
    main()
