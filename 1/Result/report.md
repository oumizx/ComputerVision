Haonan Zhu

hz1396

N19549395

 

Computer Vision Project

**1.Process**

(1) Read image and detect edge using Sobel’s detector 

Use Sobel’s detector to detect edge, get magnitude map.

(2) Normalize image

Use ![img](file:////Users/michael/Library/Group%20Containers/UBF8T346G9.Office/msoclip1/01/clip_image001.png) to normalize the grayscale value and let itlie on [0, 255]

(3) Do non-maxima suppression

Do non-maxima suppression to thin the edges.

(4) Produce binary edge map

Choose a threshold, if the grayscale of the pixel is smallerthan the threshold, set it to 0, else set it to 255.

(5) Hough Transform

Do Hough Transform and get accumulator of ![img](file:////Users/michael/Library/Group%20Containers/UBF8T346G9.Office/msoclip1/01/clip_image002.png).

(6) Filter accumulator

Set a threshold(e.g. 0.25). If the votes of related ![img](file:////Users/michael/Library/Group%20Containers/UBF8T346G9.Office/msoclip1/01/clip_image003.png) is smaller than threshold * maxVotes, filterout it.

(7) Merge lines

Since ![img](file:////Users/michael/Library/Group%20Containers/UBF8T346G9.Office/msoclip1/01/clip_image004.png) with little difference will generate sameresult, we needed to merge these lines. When the difference of ![img](file:////Users/michael/Library/Group%20Containers/UBF8T346G9.Office/msoclip1/01/clip_image005.png) and ![img](file:////Users/michael/Library/Group%20Containers/UBF8T346G9.Office/msoclip1/01/clip_image006.png) are below a threshold between lines, theselines will be merged. (e.g. ![img](file:////Users/michael/Library/Group%20Containers/UBF8T346G9.Office/msoclip1/01/clip_image007.png) difference is smaller than 2% of the range, ![img](file:////Users/michael/Library/Group%20Containers/UBF8T346G9.Office/msoclip1/01/clip_image008.png) diffrence is smaller than 3 degrees)

(8) Find parallel lines pair

![img](file:////Users/michael/Library/Group%20Containers/UBF8T346G9.Office/msoclip1/01/clip_image010.png)

![img](file:////Users/michael/Library/Group%20Containers/UBF8T346G9.Office/msoclip1/01/clip_image011.png) is theta difference threshold between twolines. C is number of votes. ![img](file:////Users/michael/Library/Group%20Containers/UBF8T346G9.Office/msoclip1/01/clip_image012.png) is threshold of votes. e.g. ![img](file:////Users/michael/Library/Group%20Containers/UBF8T346G9.Office/msoclip1/01/clip_image013.png)

If two lines fits these two equations, then these two linescan form into a pair.

(9) Match parallel lines pairs

![img](file:////Users/michael/Library/Group%20Containers/UBF8T346G9.Office/msoclip1/01/clip_image015.png)

The vertical distances (ρ axis) between peaks within eachpair are the orthogonal distances between parallel sides,

i.e., d 1 = |ρ 1 − ρ 2 | = b sin α and d 2 = |ρ 3 − ρ 4 | =a sin α. Thus, |ρ 1 − ρ 2 | = C(ρ 3 , θ 3 ) sin α = C(ρ 4 , θ 4 ) sin α and |ρ3 − ρ 4 | = C(ρ 1 , θ 1 ) sin α = C(ρ 2 , θ 2 ) sin α.

![img](file:////Users/michael/Library/Group%20Containers/UBF8T346G9.Office/msoclip1/01/clip_image017.png)

![img](file:////Users/michael/Library/Group%20Containers/UBF8T346G9.Office/msoclip1/01/clip_image019.png)

Two pairs satisfying the conditions can form a parallelogram.![img](file:////Users/michael/Library/Group%20Containers/UBF8T346G9.Office/msoclip1/01/clip_image020.png) is threshold. It should be approximately equalto ![img](file:////Users/michael/Library/Group%20Containers/UBF8T346G9.Office/msoclip1/01/clip_image021.png).

(10) Get intersections of parallel line pairs

![img](file:////Users/michael/Library/Group%20Containers/UBF8T346G9.Office/msoclip1/01/clip_image022.png) ![img](file:////Users/michael/Library/Group%20Containers/UBF8T346G9.Office/msoclip1/01/clip_image023.png) ![img](file:////Users/michael/Library/Group%20Containers/UBF8T346G9.Office/msoclip1/01/clip_image024.png)

Solve ![img](file:////Users/michael/Library/Group%20Containers/UBF8T346G9.Office/msoclip1/01/clip_image025.png).

Calculate the intersection of parallel line pairs. Filterout the intersections not in the image.

(11) Sort intersections by angles

Calculate the center point of four intersections and sortintersection based on the angle of the line generated by intersection andcenter point. It is for locating the sides of parallelogram. Just loop in orderwe will not meet diagram.

(12) Filter intersections

Calculate the fours sides’ length. If the shorted side’slength is smaller than 1/20 of image width, we filter out the intersections.

Check the side line of parallelogram based on two intersectionpoints on binary edge map. Divide the line to many separate points. Record thepoints positions on the line. If one of the pixel on 3 x 3 range of theposition on edge map is greater than 0, then add one on valid points. Finallywe calculate validPoints / totalPoints on four side lines. If four sidespercentages are all greater than a threshold value, we will pick theseintersections.

(13) Merge close parallelograms

If parallelograms’ four intersections’ related distance areall below 10, there parallelograms will be merged into one.

(14) Plot parallelogram

Plot side lines of parallelogram based on the intersections.Just plot the intersection in order because we sorted it before.

 

**2. Programminglanguage and instructions**

I use python as my programming language. Since I usedifferent parameters on different figures, I attached three separate files for testingthe figures. To test you just need to run the python files. Make sure that theorigin figures are on the same directory of the python files.

 

**3. Codes**

Attached.

 

**4. Images**

(1) Figure 1

Original image

![img](file:////Users/michael/Library/Group%20Containers/UBF8T346G9.Office/msoclip1/01/clip_image027.png)

 

Normalized gradient magnitude

 

 

 

![img](file:////Users/michael/Library/Group%20Containers/UBF8T346G9.Office/msoclip1/01/clip_image029.png)

 

Edge map after thresholding

![img](file:////Users/michael/Library/Group%20Containers/UBF8T346G9.Office/msoclip1/01/clip_image031.png)

The threshold used is 78.

 

Four corners of the parallelogram: [246, 457], [357, 666],[670, 533], [575, 310]

 

Output image

![img](file:////Users/michael/Library/Group%20Containers/UBF8T346G9.Office/msoclip1/01/clip_image033.png)

 

(2) Figure 2

Original image

![img](file:////Users/michael/Library/Group%20Containers/UBF8T346G9.Office/msoclip1/01/clip_image035.png)

 

Normalized gradient magnitude

![img](file:////Users/michael/Library/Group%20Containers/UBF8T346G9.Office/msoclip1/01/clip_image037.png)

 

Edge map after thresholding

![img](file:////Users/michael/Library/Group%20Containers/UBF8T346G9.Office/msoclip1/01/clip_image039.png)

I used non-maxima suppression before thresholding.

The threshold used is 22.

Four corners of the parallelogram: 

[572, 626], [626, 662], [871, 478], [817, 448]

[451, 502], [447, 609], [605, 380], [614, 282]

 

Output image

![img](file:////Users/michael/Library/Group%20Containers/UBF8T346G9.Office/msoclip1/01/clip_image041.png)

 

(3) Figure 3

original image

![img](file:////Users/michael/Library/Group%20Containers/UBF8T346G9.Office/msoclip1/01/clip_image043.png)

 

Normalized gradient magnitude

![img](file:////Users/michael/Library/Group%20Containers/UBF8T346G9.Office/msoclip1/01/clip_image045.png)

 

Edge map after thresholding

![img](file:////Users/michael/Library/Group%20Containers/UBF8T346G9.Office/msoclip1/01/clip_image047.png)

The threshold is 20.

 

Four corners of the parallelogram:

[115, 115], [117, 225], [230, 227], [230, 123]

[115, 115], [117, 225], [279, 228], [279, 127]

[116, 115], [114, 225], [230, 227], [230, 123]

[116, 115], [114, 225], [279, 228], [279, 127]

[170, 119], [168, 226], [279, 228], [279, 127]

 

Output image

![img](file:////Users/michael/Library/Group%20Containers/UBF8T346G9.Office/msoclip1/01/clip_image049.png)