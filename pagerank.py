import os
import random
import re
import sys


DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    transition = dict()
    # (1-damping_factor) probability because we choose randomly among all pages.
    for pages in corpus:
        transition[pages] = 1 - damping_factor
        if pages in corpus[page]:
            N = len(corpus[page])
            transition[pages] += damping_factor / N
    return transition
    

def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    sample = {page: 0 for page in corpus}
    current_page = random.choice(list(sample.keys()))
    for _ in range(n):
        sample[current_page] += 1 / n

        model_transition = transition_model(corpus, current_page,damping_factor)
        current_page = random.choice(list(model_transition.keys()))
    return sample

def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    all_pages = set()
    inlink_map = {}
    outlink_counts = {}

    for page in corpus: add_node(page, inlink_map, outlink_counts)
    for page in corpus:
        outlink_counts[page] = len(corpus[page])
        all_pages.add(page)
        for item in corpus[page]: 
            inlink_map[item].add(page)
            all_pages.add(item)

    rank = {page:1/(len(all_pages)) for page in corpus} 
    new_rank = dict()
    while True:
        for page, links in corpus.items():
             new_rank[page] = ((1 - damping_factor) / len(corpus))  + (damping_factor * sum(rank[inlink] \
                         / outlink_counts[inlink] for inlink in links))

        new_rank, rank = rank, new_rank
        

        if evaluate(new_rank, rank): return rank



def evaluate(dic1, dic2):
    for u, v in zip(dic1, dic2):
        if abs(dic1[u] - dic2[v]) > .001 : return False   
    return True

def add_node(node, inlink, outlink):
    if node not in inlink: inlink[node] = set()
    if node not in outlink: outlink[node] = 0

if __name__ == "__main__":
    main()
