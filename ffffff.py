from pdf2image import convert_from_path

pdf_path = "D:\SIH 2023\malayalam.pdf"

images = convert_from_path(pdf_path)

for i, image in enumerate(images):
    image.save(f"page_{i + 1}.png")
