# The r/Jokes Dataset: a Large Scale Humor Collection
Code and Datasets from the paper, ["The r/Jokes Dataset: a Large Scale Humor Collection"](https://www.aclweb.org/anthology/2020.lrec-1.753/) by Orion Weller and Kevin Seppi

Dataset files are located in `data/{train/dev/test}.tsv` for the regression task, while the full unsplit data can be found in `data/preprocessed.tsv`.  These files will need to be unzipped after cloning the repo.

For related projects, see our work on [Humor Detection (separating the humorous jokes from the non-humorous)](https://github.com/orionw/RedditHumorDetection) or [generating humor automatically](https://github.com/orionw/humorTranslate).

** **We do not endorse these jokes. Please view at your own risk** **

## License
The data is under the [Reddit License and Terms of Service](https://www.reddit.com/wiki/api-terms) and users must follow the Reddit User Agreement and Privacy Policy, as well as remove any posts if asked to by the original user.  For more details on this, please see the link above.  

# Usage
## Load the Required Packages
0. Run `pip3 install -r requirements.txt`
1. Gather the NLTK packages by running `bash download_nltk_packages.sh`.  This downloads the packages `averaged_perceptron_tagger`, `words`, `stopwords`, `maxent_ne_chunker`, used for analysis/preprocessing.

## Reproduce the current dataset (updated to Jan 1st 2020)
### We chunk this process into three parts to avoid networking errors
0. Run `python3 gather_reddit_pushshift.py` after `cd prepare_data` to gather the Reddit post ids.
1. Run `python3 preprocess.py --update` to update the Reddit post IDs with the full post.
2. Run `python3 preprocess.py --preprocess` to preprocess the Reddit posts into final datasets

## Reproduce plots and analysis from the paper
0. Run `cd analysis`
1. Run `python3 time_statistics.py` to gather the statistics that display over time
2. Run `python3 dataset_statistics.py` to gather the overall dataset statistics
3. See plots in the `./plots` folder

## Re-gather All Jokes and Extend With Newer Jokes 
0. Run the first two commands in the `Reproduce` section above
1. Update the code in the `preprocess` function of the `preprocess.py` file to NOT remove all jokes after 2020 (line 89).  Then run `python3 preprocess.py --preprocess`

# Reference:
If you found this repository helpful, please cite the following paper:
```
@ARTICLE{rjokesData2020,
  title={The r/Jokes Dataset: a Large Scale Humor Collection},
  author={Weller, Orion and Seppi, Kevin},
  journal={"Proceedings of the 2020 Conference of Language Resources and Evaluation"},
  month=May,
  year = "2020",
}
