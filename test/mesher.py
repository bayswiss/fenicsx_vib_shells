import gmsh
 
gmsh.initialize()
gmsh.model.add("square")


gmsh.option.setNumber("Mesh.MeshSizeMin", 0.02)
gmsh.option.setNumber("Mesh.MeshSizeMax", 0.02)
# addRectangle(x, y, z, dx, dy) — one line!
gmsh.model.occ.addRectangle(0, 0, 0, 0.3, 0.3)


gmsh.model.occ.synchronize()
# gmsh.fltk.run()
gmsh.option.setNumber("Mesh.RecombineAll", 1)
gmsh.option.setNumber("Mesh.Algorithm", 8)
gmsh.model.addPhysicalGroup(2, [1], 1, "structure")
gmsh.model.addPhysicalGroup(1, [1,2,3,4], 2, "bc")
gmsh.model.mesh.generate(2)
gmsh.fltk.run()

gmsh.write("square_quads.msh")
 
gmsh.finalize()