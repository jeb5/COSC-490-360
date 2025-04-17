import spatialmedia
import argparse
import spatialmedia.metadata_utils

def addSphericalMetadata(input_path, output_path):
  metadata = spatialmedia.metadata_utils.Metadata()
  metadata.video = spatialmedia.metadata_utils.generate_spherical_xml()
  def logging(message): print(message)
  spatialmedia.metadata_utils.inject_metadata(input_path, output_path, metadata, logging)

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Add spherical metadata to a video file.")
	parser.add_argument("input_path", type=str, help="Path to the input video file.")
	parser.add_argument("output_path", type=str, help="Path to the output video file.")
	args = parser.parse_args()

	addSphericalMetadata(args.input_path, args.output_path)
  