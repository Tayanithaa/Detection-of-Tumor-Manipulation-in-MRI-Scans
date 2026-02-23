from tampering import generate_manipulated

generate_manipulated(
    "../Dataset/Processed/train/real",
    "../Dataset/Processed/train/manipulated"
)

generate_manipulated(
    "../Dataset/Processed/test/real",
    "../Dataset/Processed/test/manipulated"
)