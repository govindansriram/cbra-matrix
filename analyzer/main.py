from dot_product import DotProductGraph
import sys


if __name__ == '__main__':
    args = sys.argv[1:]

    if args[0] == "dp":
        graph = DotProductGraph(args[1:])
        graph.pipeline()