## Инструкция по запуску
* **Запустите roscore**
<pre>$ roscore</pre>
* **Запустите окружение turtlesim_node**
<pre>$ rosrun turtlesim turtlesim_node</pre>
* **Заспауните вторую черепашку**
<pre>$ rosservice call /spawn "x:0.0
y:5.0
theta:0.0
name:'micheliangelo'"
</pre>
* **Соберите workspace**
<pre>SDC/turtles_ws$ catkin_make</pre>
* **Запустите node с заданием**
<pre>$ rosrun turtles turtle.py</pre>

## Как проверить
Проверить, что всё работает можно, запустив node, с помощью которой можно управлять первой черепашкой
<pre>$ rosrun turtlesim turtle_teleop_key</pre>
