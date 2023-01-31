from PIL import Image
  
image = Image.open('image072.png')
  
right = 100
left = 100
top = 100
bottom = 100
  
width, height = image.size
  
new_width = width + right + left
new_height = height + top + bottom
  
result = Image.new(image.mode, (new_width, new_height), (0, 255, 255))
  
result.paste(image, (left, top))
  
result.save('image072_padding.png')