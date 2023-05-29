from svg_turtle import SvgTurtle
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM
import cairosvg
import turtle


cpp_keywords = ['while', 'break', 'else' ,'if', 'for', 'int', 'double', 'char', 'new', 'void']

class Turtle:
    T = SvgTurtle(1024, 1024)
    T.speed(0)

    
    angle = 360 // len(cpp_keywords)

    # 원점과 keywords 노드 사이의 거리 default : 200
    length = 200
    # 그리는 원의 size default : 20
    size = 20
    color = {
        'r':0.0,
        'g':0.0,
        'b':0.0
    }

    def __init__(self):
        self.T.clear()
        self.size = 20
        self.color = {
        'r':0.0,
        'g':0.0,
        'b':0.0
        }
        self.length = 200
        self.angle = 360 // len(cpp_keywords)

    keywordsPosition = {}

    def setKeywordsPosition(self, on):
        self.T.penup()
        for _ in range(len(cpp_keywords)):
            self.T.home()
            self.T.setheading(self.angle * _)

            self.T.forward(self.length)
            
            if on:
                self.T.write(cpp_keywords[_], font=('Arial', 20, 'bold'))
            self.keywordsPosition[cpp_keywords[_]] = [self.T.xcor(), self.T.ycor()]
        self.T.pendown()

    def rgb_to_hex(self, r, g, b):
            r = int(r)
            g = int(g)
            b = int(b)
            return "#{:02x}{:02x}{:02x}".format(r, g, b)

    def drawLine(self, prev, next):
        if(prev != next):
            self.T.penup()
            self.T.setposition(self.keywordsPosition[prev][0], self.keywordsPosition[prev][1])
            self.T.pendown()
            self.T.goto(self.keywordsPosition[next][0], self.keywordsPosition[next][1])
        elif(prev == next):
            self.T.penup()
            self.T.setposition(self.keywordsPosition[prev][0], self.keywordsPosition[prev][1])
            self.T.pendown()
            self.T.circle(self.size)
            self.size = self.size + 4
        
        self.T.pencolor(self.rgb_to_hex(self.color['r'], self.color['g'], self.color['b']))

        self.color['r'] += 2

        

    def done(self, file):
        self.T.save_as("Image/{}.svg".format(file))
        drawing = svg2rlg("Image/{}.svg".format(file))
        # renderPM.drawToFile(drawing, "image/{}.png".format(file), fmt="PNG")
        
        input_file = './Image/' + file + '.svg'
        output_file = './Image/' + file + '.png'
        cairosvg.svg2png(url=input_file, write_to=output_file)
        # print(file)