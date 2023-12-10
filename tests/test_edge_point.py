w = 600
h = 400
start = (300, 200)

#### top right

end = (450, 100)
find_edge_intersection(w, h, start, end) # (599, 0) Q1 diagonal

end  = (450, 150)
find_edge_intersection(w, h, start, end) # (599, 100) Q1 flat

end = (450, 50)
find_edge_intersection(w, h, start, end) # (500, 0) Q1 tall

print()

###### bottom right
end = (450, 300)
find_edge_intersection(w, h, start, end) # (599, 399) 

end = (450, 350)
find_edge_intersection(w, h, start, end) # (500, 400)

end = (450, 250)
find_edge_intersection(w, h, start, end) # (600, 300)

print()

###### top left
end = (150, 100)
find_edge_intersection(w, h, start, end) # (0, 0)

end  = (150, 150)
find_edge_intersection(w, h, start, end) # (0, 100)

end = (150, 50)
find_edge_intersection(w, h, start, end) # (100, 0)

print()

###### bottom left
end = (150, 300)
find_edge_intersection(w, h, start, end) # (0, 399)

end  = (150, 350)
find_edge_intersection(w, h, start, end) # (101, 399)

end = (150, 250)
find_edge_intersection(w, h, start, end) # (0, 300)

print()