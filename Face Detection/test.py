from face_embedder import FaceEmbedder

embedder = FaceEmbedder()

sim, match = embedder.compare("/Users/sanchitsuman/vcs/github.com/drcscodes/ML-Projects/Face Detection/faces/image4.png",
                                     "/Users/sanchitsuman/vcs/github.com/drcscodes/ML-Projects/Face Detection/faces/image3.png")

print("Similarity:", sim)
print("Match:", match)