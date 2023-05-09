
from tkinter import *

root = Tk()
canvas = Canvas(root, width=500, height=500)
canvas.pack()

# 绘制美羊羊轮廓
canvas.create_oval(150, 150, 350, 350, fill="#F6D5E6")

# 绘制美羊羊眼睛
canvas.create_oval(200, 200, 225, 225, fill="white", outline="black")
canvas.create_oval(275, 200, 300, 225, fill="white", outline="black")
canvas.create_oval(212, 212, 218, 218, fill="black")
canvas.create_oval(287, 212, 293, 218, fill="black")

# 绘制美羊羊耳朵
canvas.create_oval(130, 190, 170, 230, fill="#F6D5E6")
canvas.create_oval(330, 190, 370, 230, fill="#F6D5E6")

# 绘制美羊羊嘴巴
canvas.create_arc(200, 275, 300, 325, start=180, extent=180, fill="#F6D5E6", outline="black")

root.mainloop()