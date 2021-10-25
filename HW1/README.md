## Инструкция по запуску
* **Запустите roscore**
<pre>$ roscore</pre>

* **Запустите rviz**
<pre>SDC/HW1$ rviz -d laser.rviz</pre>

* **Соберите workspace**
<pre>SDC/HW1$ catkin_make</pre>
<pre>SDC/HW1$ source devel/setup.bash</pre>

* **Запустите node с заданием**
<pre>$ rosrun laser track_laser.py</pre>

## Как проверить
Проверить, что всё работает можно, запустив bag-файл, после чего в rviz начнут отображаться маркеры и карта
<pre>$ rosbag play 2011-01-25-06-29-26.bag</pre>
