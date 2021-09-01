
def generate_lines_that_include(string, fp):
  for line in fp:
    if (line.find(string) != -1):
      yield line
 
data_file = open('data/tweets.txt', 'w', newline='')
 
with  open('data/tweet.js', 'r') as td:
  for line in generate_lines_that_include("full_text", td):
    filtered = line.replace("      \"full_text\" : ", '')
    output = filtered.replace("\n", '')
    final = output[1:-1-2] #strip all end spaces and commas
    print(final)
    print(final, file=data_file)

data_file.close() 
