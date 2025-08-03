import argparse


def main(args):
  pass


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Process video+inertials to produce 360 video")
  parser.add_argument("directory", type=str, help="Path to directory containing video+inertials")
  parser.add_argument("--produce_debug", action="store_true")
  parser.add_argument("--produce_360", action="store_true")
  parser.add_argument("--use_features_cache", action="store_true")
  parser.add_argument("--use_matches_cache", action="store_true")
  parser.add_argument("--window_size", type=int, default=1)
  parser.add_argument("--window_strategy", type=str, choices=["simple", "quadratic", "overlapping"], default="simple")
  parser.add_argument("--output_scale", type=float, default=1.0)

  args = parser.parse_args()
  main(args)
