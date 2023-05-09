import tkinter as tk

root = tk.Tk()
root.title("美羊羊")

canvas = tk.Canvas(root, width=300, height=400)
canvas.pack()

# 绘制身体
body = canvas.create_oval(100, 150, 200, 300, fill='white', outline='black')
# 绘制脚
foot1 = canvas.create_oval(100, 275, 130, 305, fill='white', outline='black')
foot2 = canvas.create_oval(170, 275, 200, 305, fill='white', outline='black')
# 绘制臂
arm1 = canvas.create_oval(65, 200, 95, 230, fill='white', outline='black')
arm2 = canvas.create_oval(205, 200, 235, 230, fill='white', outline='black')
# 绘制头
head = canvas.create_oval(110, 50, 190, 150, fill='white', outline='black')
# 绘制眼睛
eye1 = canvas.create_oval(130, 70, 150, 90, fill='black', outline='black')
eye2 = canvas.create_oval(170, 70, 190, 90, fill='black', outline='black')
# 绘制嘴巴
mouth = canvas.create_arc(130, 110, 170, 130, start=-180, extent=180, fill='white', outline='black')

# 运行窗口
root.mainloop()
