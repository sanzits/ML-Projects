from compare_faces import CompareFaces

matcher = CompareFaces("face_embedding_model.keras")

score, match = matcher.compare_faces("/Users/sanchitsuman/vcs/github.com/drcscodes/ML-Projects/Face Detection/faces/image2.png",
                                     "/Users/sanchitsuman/vcs/github.com/drcscodes/ML-Projects/Face Detection/faces/image2.png")

print("Distance:", score)
print("Match:", match)


