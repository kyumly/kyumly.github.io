---
layout : single-2
title : "cs231n 강의 노트 4장 Backpropagation, Intuitions"
description : "Backpropagation, Intuitions"
categories : cs231
tag : [python, AI, Note]
toc : true
toc_sticky : true
author_profile : false
use_math : true
---


<h1>cs231n 강의 노트 4장 Backpropagation, Intuitions</h1> <br>


# 1. Introduction

## 1.1 Motivation
In this section we will develop expertise with an intuitive understanding of **backpropagation**, which is a way of computing gradients of expressions through recursive application of **chain rule**.<br>
Understanding of this process and its subtleties is critical for you to understand, and effectively develop, design and debug neural networks.<br>

- 이 섹션에서 우리는 체인 규칙의 재귀적인 적용을 통해 그레이디언트를 계산하는 방법인 역전파에 대한 직관적인 이해를 가진 전문 지식을 개발할 것입니다. 
- 이 프로세스와 프로세스의 세부 사항을 이해하는 것은 신경망을 이해하고 효과적으로 설계 및 디버그하는 데 매우 중요합니다.

- 요약 :  Gradient 직관적으로 개발할 것이고, 역전파를 알아야 효과적으로 설계 및 디버기를 할 수 있다.

## 1.2 Problem statement
The core problem studied in this section is as follows:<br>
We are given some function \\(f(x)\\) where \\(x\\) is a vector of inputs and we are interested in computing the gradient of \\(f\\) at \\(x\\) (i.e. \\(\nabla f(x)\\) )

- 이 섹션에서 연구하는 핵심 문제는 다음과 같습니다. 
- \\(x\\)가 입력 벡터인 함수 \\(f(x)\\)가 주어지고 \\(x\\) 에서 \\(f(x)\\)의 기울기를 계산하는 데 관심이 있습니다(i.e. \\(\nabla f(x)\\) )


## 1.3 Motivation.
Recall that the primary reason we are interested in this problem is that in the specific case of neural networks, \\(f\\) will correspond to the loss function ( \\(L\\) ) and the inputs \\(x\\) will consist of the training data and the neural network weights.<br>
For example, the loss could be the SVM loss function and the inputs are both the training data \\((x_i,y_i), i=1 \ldots N\\) and the weights and biases \\(W,b\\). <br>
Note that (as is usually the case in Machine Learning) we think of the training data as given and fixed, and of the weights as variables we have control over.<br>
Hence, even though we can easily use backpropagation to compute the gradient on the input examples \\(x_i\\), in practice we usually only compute the gradient for the parameters (e.g. \\(W,b\\)) so that we can use it to perform a parameter update. <br>
However, as we will see later in the class the gradient on \\(x_i\\) can still be useful sometimes, for example for purposes of visualization and interpreting what the Neural Network might be doing.

- 우리가 이 문제에 관심을 갖는 주된 이유는 신경망의 특정 경우에서 \\(f\\)가 손실 함수( \\(L\\) )에 해당하고 입력 \\(x\\)가 훈련 데이터와 신경망 가중치로 구성된다는 점을 상기하십시오. 
- 예를 들어, 손실은 SVM 손실 함수일 수 있고 입력은 훈련 데이터 \\((x_i,y_i), i=1 \ldots N\\) 과 가중치 및 편향 \\(W,b\\)입니다.
- (머신 러닝에서 일반적으로 그렇듯이) 우리는 훈련 데이터를 주어진 고정(fixed)된 것으로 생각하고 가중치를 우리가 통제(control)할 수 있는 변수로 생각합니다.
- 따라서 역전파를 사용하여 입력 예제 \\(x_i\\)에 대한 그래디언트를 쉽게 계산할 수 있지만, 최적의 파라미터를 사용하기 위해서는 실제로는 사용하는 파라미터인(예:  \\(W,b\\))) 대한 그래디언트만 계산합니다. 
- 그러나 클래스의 후반부에서 볼 수 있듯이 \\(x_i\\)의 그래디언트는 예를 들어 신경망이 수행할 수 있는 작업을 시각화하고 해석하는 목적으로 여전히 유용할 수 있습니다.
  
- **정리** :\\(x\\) 대한 gradient 구하는 것이 아니라 W,b에 대한 파라미터를 활용하여 역전파를 실행한다.

If you are coming to this class and you're comfortable with deriving gradients with chain rule, we would still like to encourage you to at least skim this section,<br>
since it presents a rarely developed view of backpropagation as backward flow in real-valued circuits and any insights you'll gain may help you throughout the class.<br>

- 이 수업에 참석하고 체인 규칙을 사용하여 그래디언트를 유도하는 데 익숙하다면 이 섹션을 최소한 훑어볼 것을 권장합니다. 
- 이 섹션은 실수값 회로에서 역방향 흐름으로 역전파에 대한 거의 개발되지 않은 보기를 제공하기 때문입니다. 그리고 당신이 얻게 될 통찰력은 수업 내내 당신에게 도움이 될 수 있습니다.

# 2.[Simple expressions, interpreting the gradient](https://cs231n.github.io/optimization-2/#grad)


Lets start simple so that we can develop the notation and conventions for more complex expressions. <br>
Consider a simple multiplication function of two numbers \\(f(x,y) = x y\\). It is a matter of simple calculus to derive the partial derivative for either input:

- 더 복잡한 표현을 위한 표기법과 규칙을 개발할 수 있도록 간단하게 시작하겠습니다.
- 두 숫자 \\(f(x,y) = x y\\).의 단순 곱셈 함수를 생각해 보십시오.<br> 
- 두 입력 중 하나에 대한 부분 미분을 도출하는 것은 간단한 미적분학의 문제입니다:

$$
f(x,y) = x y \hspace{0.5in} \rightarrow \hspace{0.5in} \frac{\partial f}{\partial x} = y \hspace{0.5in} \frac{\partial f}{\partial y} = x
$$


Interpretation. Keep in mind what the derivatives tell you: They<br> indicate the rate of change of a function with respect to that variable surrounding an infinitesimally small region near a particular point:

-   도함수가 말하는 것을 명심하십시오. 도함수는 특정 지점 근처의 극히 작은 영역을 둘러싼 변수에 대한 함수의 변화율을 나타냅니다.

$$
\frac{df(x)}{dx} = \lim_{h\ \to 0} \frac{f(x + h) - f(x)}{h}
$$


A technical note is that the division sign on the left-hand side is, unlike the division sign on the right-hand side, not a division.<br>
Instead, this notation indicates that the operator \\(  \frac{d}{dx} \\) is being applied to the function \\(f\\), and returns a different function (the derivative).<br>
A nice way to think about the expression above is that when \\(h\\) is very small, then the function is well-approximated by a straight line, and the derivative is its slope.<br>
In other words, the derivative on each variable tells you the sensitivity of the whole expression on its value.<br>
For example, if \\(x = 4, y = -3\\) then \\(f(x,y) = -12\\) and the derivative on \\(x\\) \\(\frac{\partial f}{\partial x} = -3\\).<br> 
This tells us that if we were to increase the value of this variable by a tiny amount, the effect on the whole expression would be to decrease it (due to the negative sign), and by three times that amount.<br> 
This can be seen by rearranging the above equation ( \\( f(x + h) = f(x) + h \frac{df(x)}{dx} \\) ). Analogously, since \\(\frac{\partial f}{\partial y} = 4\\), we expect that increasing the value of \\(y\\) by some very small amount \\(h\\) would also increase the output of the function (due to the positive sign), and by \\(4h\\).<br>

- 기술적인 참고 사항은 왼쪽에 있는 나눗셈 기호는 오른쪽에 있는 나눗셈 기호와 달리 나눗셈 기호가 아니라는 것입니다.
- 대신, 이 표기법은 연산자 \\(  \frac{d}{dx} \\) 가 함수 \\(f\\)에 적용되고 있음을 나타내며, 다른 함수(도함수)를 반환합니다.
- 위의 식에 대해 생각해 볼 수 있는 좋은 방법은 \\(h\\)가 매우 작을 때 함수가 직선으로 잘 근사되고 도함수가 기울기라는 것입니다.
- 즉, 각 변수의 미분은 값에 대한 전체 식의 민감도를 알려줍니다. 
- 예를 들어,\\(x = 4, y = -3\\)이면 \\(f(x,y) = -12\\)이고 \\(x\\)의 \\(\frac{\partial f}{\partial x} = -3\\) 도함수입니다. 
- 이것은 만약 우리가 이 변수의 값을 아주 작은 양만큼 증가시킨다면, 전체 표현식에 대한 효과는 그것을 (부정적인 부호로 인해) 감소시킬 것이고, 그 양의 3배가 될 것이라는 것을 말해줍니다.
- 이는 위의 방정식( \\( f(x + h) = f(x) + h \frac{df(x)}{dx} \\) )을 재정렬하여 확인할 수 있습니다. 마찬가지로 \\(\frac{\partial f}{\partial y} = 4\\)이므로 \\(y\\)  값을 매우 작은 양 \\(h\\)만큼 증가시키면 함수의 출력도 \\(4h\\)만큼 증가할것으로 예상합니다


The derivative on each variable tells you the sensitivity of the whole expression on its value.
-   각 변수의 도함수는 해당 값에 대한 전체 표현식의 민감도를 알려줍니다.

As mentioned, the gradient \\(\nabla f\\) is the vector of partial derivatives, so we have that \\(\nabla f = [\frac{\partial f}{\partial x}, \frac{\partial f}{\partial y}] = [y, x]\\). <br>
Even though the gradient is technically a vector, we will often use terms such as *"the gradient on x"* instead of the technically correct phrase *"the partial derivative on x"* for simplicity.<br>

We can also derive the derivatives for the addition operation:

- 언급한 바와 같이 기울기 ∇f는 부분 도함수의 벡터이므로 \\(\nabla f = [\frac{\partial f}{\partial x}, \frac{\partial f}{\partial y}] = [y, x]\\)가 됩니다.
- 그래디언트는 기술적으로는 벡터이지만 단순화를 위해 기술적으로 올바른 문구인 \"x의 편도함수\" 대신 \"x의 그래디언트\"와 같은 용어를 자주 사용합니다.

- 덧셈 연산에 대한 도함수를 도출할 수도 있습니다.

$$
f(x,y) = x + y \hspace{0.5in} \rightarrow \hspace{0.5in} \frac{\partial f}{\partial x} = 1 \hspace{0.5in} \frac{\partial f}{\partial y} = 1
$$

that is, the derivative on both \\(x,y\\) is one regardless of what the values of \\(x,y\\) are.<br>
This makes sense, since increasing either \\(x,y\\) would increase the output of \\(f\\), and the rate of that increase would be independent of what the actual values of \\(x,y\\) are (unlike the case of multiplication above).<br> 
The last function we'll use quite a bit in the class is the *max* operation:<br>

- 즉, x,y의 값이 무엇이든 관계없이 x,y 둘 다에 대한 도함수는 1입니다. 
- 이는 x,y 중 하나를 증가시키면 f의 출력이 증가하고 해당 증가율은 x,y의 실제 값과 무관하기 때문에 의미가 있습니다(위의 곱셈의 경우와 달리). 
- 클래스에서 꽤 많이 사용할 마지막 함수는 max 작업입니다.

$$
f(x,y) = \max(x, y) \hspace{0.5in} \rightarrow \hspace{0.5in} \frac{\partial f}{\partial x} = \mathbb{1}(x >= y) \hspace{0.5in} \frac{\partial f}{\partial y} = \mathbb{1}(y >= x)
$$

That is, the (sub)gradient is 1 on the input that was larger and 0 on the other input. <br>
Intuitively, if the inputs are \\(x = 4,y = 2\\), then the max is 4, and the function is not sensitive to the setting of \\(y\\).<br> 
That is, if we were to increase it by a tiny amount \\(h\\), the function would keep outputting 4, and therefore the gradient is zero: there is no effect.<br> 
Of course, if we were to change \\(y\\) by a large amount (e.g. larger than 2), then the value of \\(f\\) would change, but the derivatives tell us nothing about the effect of such large changes on the inputs of a function;<br> 
They are only informative for tiny, infinitesimally small changes on the inputs, as indicated by the \\(\lim_{h \rightarrow 0}\\) in its definition.<br>

- 즉, (서브)그라데이션은 더 큰 입력에서 1이고 다른 입력에서는 0입니다. 
- 직관적으로 입력이 x=4,y=2이면 최대값은 4이고, 함수는 y 영향을 받지 않습니다.
- 즉, 우리가 그것을 아주 작은 양 h만큼 증가시킨다면, 함수는 계속해서 4를 출력할 것이고, 따라서 기울기는 0입니다: 효과가 없습니다.
- 물론 y를 크게(예: 2보다 크게) 변경하면 f의 값이 변경되지만 도함수는 함수의 입력에 대한 이러한 큰 변경의 영향에 대해 아무 것도 알려주지 않습니다. 
- 그것들은 정의에서 lim h→0으로 표시된 것처럼 입력에 대한 아주 작고 극소의 작은 변화에 대해서만 유익합니다.

# 3. Compound expressions with chain rule

Lets now start to consider more complicated expressions that involve multiple composed functions, such as \\(f(x,y,z) = (x + y) z\\).<br>
This expression is still simple enough to differentiate directly, but we'll take a particular approach to it that will be helpful with understanding the intuition behind backpropagation.<br>
In particular, note that this expression can be broken down into two expressions: \\(q = x + y\\) and \\(f = q z\\). <br>
Moreover, we know how to compute the derivatives of both expressions separately, as seen in the previous section. <br>
\\(f\\) is just multiplication of \\(q\\) and \\(z\\), so \\(\frac{\partial f}{\partial q} = z, \frac{\partial f}{\partial z} = q\\), and \\(q\\) is addition of \\(x\\) and \\(y\\) so \\( \frac{\partial q}{\partial x} = 1, \frac{\partial q}{\partial y} = 1 \\).<br>
However, we don't necessarily care about the gradient on the intermediate value \\(q\\) - the value of \\(\frac{\partial f}{\partial q}\\) is not useful. <br>
Instead, we are ultimately interested in the gradient of \\(f\\) with respect to its inputs \\(x,y,z\\). The **chain rule** tells us that the correct way to "chain" these gradient expressions together is through multiplication. For example, \\(\frac{\partial f}{\partial x} = \frac{\partial f}{\partial q} \frac{\partial q}{\partial x} \\). In practice this is simply a multiplication of the two numbers that hold the two gradients. Lets see this with an example:<br>

- 이제 \\(f(x,y,z) = (x + y) z\\)와 같이 여러 구성 함수를 포함하는 더 복잡한 식을 고려해 보겠습니다.
- 이 표현은 여전히 직접적으로 구별할 수 있을 정도로 간단하지만, 역전파의 직관을 이해하는 데 도움이 될 특정한 접근 방식을 취할 것입니다.
- 특히 이 표현식은 \\(q = x + y\\) 그리고 \\(f = q z\\)의 두 가지 표현식으로 나눌 수 있습니다.
- 또한, 우리는 이전 섹션에서 본 것처럼 두 표현식의 파생물을 개별적으로 계산하는 방법을 알고 있습니다.
- \\(f\\)는 \\(q\\) 그리고 \\(z\\)의 곱일 뿐이므로 \\(\frac{\partial f}{\partial q} = z, \frac{\partial f}{\partial z} = q\\), \\(q\\)는 \\(x\\)와 \\(y\\)의 덧셈이므로 \\( \frac{\partial q}{\partial x} = 1, \frac{\partial q}{\partial y} = 1 \\)입니다. 
- 그러나 중간 값 \\(q\\)에 대한 기울기에 대해 반드시 관심이 있는 것은 아닙니다. \\(\frac{\partial f}{\partial q}\\)의 값은 유용하지 않습니다.
- 대신, 우리는 궁극적으로 입력 x,y,z에 대한 f의 기울기 에 관심이 있습니다. 
- 체인 규칙은 이러한 그레이디언트 표현을 함께 \"체인\"하는 올바른 방법이 곱셈을 통해 있음을 알려줍니다. 
- 예를 들어, ∂f∂x=∂f/∂q \* ∂q/∂x입니다. 실제로 이것은 두 그라데이션을 유지하는 두 숫자의 단순한 곱입니다. 
- 예를 들어 이를 살펴보겠습니다:

```python
# set some inputs
x = -2; y = 5; z = -4

# perform the forward pass
q = x + y # q becomes 3
f = q * z # f becomes -12

# perform the backward pass (backpropagation) in reverse order:
# first backprop through f = q * z
dfdz = q # df/dz = q, so gradient on z becomes 3
dfdq = z # df/dq = z, so gradient on q becomes -4
dqdx = 1.0
dqdy = 1.0
# now backprop through q = x + y
dfdx = dfdq * dqdx  # The multiplication here is the chain rule!
dfdy = dfdq * dqdy
```


We are left with the gradient in the variables `[dfdx,dfdy,dfdz]`, which tell us the sensitivity of the variables `x,y,z` on `f`!. <br>
This is the simplest example of backpropagation. <br>
Going forward, we will use a more concise notation that omits the `df` prefix.<br>
For example, we will simply write `dq` instead of `dfdq`, and always assume that the gradient is computed on the final output.<br>

This computation can also be nicely visualized with a circuit diagram:<br>

- 변수 `[dfdx,dfdy,dfdz]`에 그래디언트가 남아 있어 f에 대한 변수 x,y,z의 민감도를 알려줍니다. 
- 이것은 역전파의 가장 간단한 예입니다.
- 앞으로는 `df` 접두사를 생략하는 더 간결한 표기법을 사용할 것입니다.
- 예를 들어, `dfdq` 대신 `dq`를 간단히 작성하고 항상 그래디언트가 최종 출력에서 계산된다고 가정합니다.

-   이 계산은 회로 다이어그램으로 멋지게 시각화할 수도 있습니다.

<div class="fig figleft fighighlight" style="align-content: center">
  <svg style="max-width: 100%" viewbox="0 0 420 220"><defs><marker id="arrowhead" refX="6" refY="2" markerWidth="6" markerHeight="4" orient="auto"><path d="M 0,0 V 4 L6,2 Z"></path></marker></defs><line x1="40" y1="30" x2="110" y2="30" stroke="black" stroke-width="1"></line><text x="45" y="24" font-size="16" fill="green">-2</text><text x="45" y="47" font-size="16" fill="red">-4</text><text x="35" y="24" font-size="16" text-anchor="end" fill="black">x</text><line x1="40" y1="100" x2="110" y2="100" stroke="black" stroke-width="1"></line><text x="45" y="94" font-size="16" fill="green">5</text><text x="45" y="117" font-size="16" fill="red">-4</text><text x="35" y="94" font-size="16" text-anchor="end" fill="black">y</text><line x1="40" y1="170" x2="110" y2="170" stroke="black" stroke-width="1"></line><text x="45" y="164" font-size="16" fill="green">-4</text><text x="45" y="187" font-size="16" fill="red">3</text><text x="35" y="164" font-size="16" text-anchor="end" fill="black">z</text><line x1="210" y1="65" x2="280" y2="65" stroke="black" stroke-width="1"></line><text x="215" y="59" font-size="16" fill="green">3</text><text x="215" y="82" font-size="16" fill="red">-4</text><text x="205" y="59" font-size="16" text-anchor="end" fill="black">q</text><circle cx="170" cy="65" fill="white" stroke="black" stroke-width="1" r="20"></circle><text x="170" y="70" font-size="20" fill="black" text-anchor="middle">+</text><line x1="110" y1="30" x2="150" y2="65" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="110" y1="100" x2="150" y2="65" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="190" y1="65" x2="210" y2="65" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="380" y1="117" x2="450" y2="117" stroke="black" stroke-width="1"></line><text x="385" y="111" font-size="16" fill="green">-12</text><text x="385" y="134" font-size="16" fill="red">1</text><text x="375" y="111" font-size="16" text-anchor="end" fill="black">f</text><circle cx="340" cy="117" fill="white" stroke="black" stroke-width="1" r="20"></circle><text x="340" y="127" font-size="20" fill="black" text-anchor="middle">*</text><line x1="280" y1="65" x2="320" y2="117" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="110" y1="170" x2="320" y2="117" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="360" y1="117" x2="380" y2="117" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line></svg>
</div>


The real-valued <i>"circuit"</i> on left shows the visual representation of the computation.<br> 
The <b>forward pass</b> computes values from inputs to output (shown in green). <br>
The <b>backward pass</b> then performs backpropagation which starts at the end and recursively applies the chain rule to compute the gradients (shown in red) all the way to the inputs of the circuit.<br> 
The gradients can be thought of as flowing backwards through the circuit.<br>

- 왼쪽의 실제 값 회로는 계산의 시각적 표현을 보여줍니다. 
- 정방향 패스는 입력에서 출력으로 값을 계산합니다(녹색으로 표시됨). 
- 그런 다음 역방향 패스는 끝에서 시작하여 체인 규칙을 재귀적으로 적용하여 회로의 입력까지 그레이디언트(빨간색으로 표시)를 계산합니다.
- 구배는 회로를 통해 역류하는 것으로 생각할 수 있습니다.

# 4. [Intuitive understanding of backpropagation](https://cs231n.github.io/optimization-2/#intuitive)

Notice that backpropagation is a beautifully local process.
Every gate in a circuit diagram gets some inputs and can right away compute two things: 1. its output value and 2.the *local* gradient of its output with respect to its inputs.<br> 
Notice that the gates can do this completely independently without being aware of any of the details of the full circuit that they are embedded in. <br>
However, once the forward pass is over, during backpropagation the gate will eventually learn about the gradient of its output value on the final output of the entire circuit.<br> 
Chain rule says that the gate should take that gradient and multiply it into every gradient it normally computes for all of its inputs<br>

- 역전파는 아주 지역적인 과정입니다. 
- 회로도의 모든 게이트는 몇 가지 입력을 받고 1. 출력 값과 2. 입력에 대한 출력의 로컬 기울기의 두 가지를 즉시 계산할 수 있습니다. 
- 게이트는 내장된 전체 회로의 세부 정보를 전혀 인식하지 않고 완전히 독립적으로 이 작업을 수행할 수 있습니다. 
- 그러나, 일단 전진 패스가 끝나면, 역전파 중에 게이트는 결국 전체 회로의 최종 출력에 대한 출력 값의 기울기를 알게 됩니다. 
- 체인 규칙에 따르면 게이트는 해당 기울기를 사용하여 모든 입력에 대해 일반적으로 계산하는 모든 기울기로 곱해야 합니다.

This extra multiplication (for each input) due to the chain rule can turn a single and relatively useless gate into a cog in a complex circuit such as an entire neural network.<br>

- 체인 규칙으로 인한 이 추가 곱셈(각 입력에 대한)은 단일하고 상대적으로 쓸모없는 게이트를 전체 신경망과 같은 복잡한 회로에서 톱니바퀴로 바꿀 수 있습니다.

- 복잡한 곳에서도 chain rule 사용하면 계산이 가능하다.

Lets get an intuition for how this works by referring again to the example. <br>
The add gate received inputs [-2, 5] and computed output 3. <br>
Since the gate is computing the addition operation, its local gradient for both of its inputs is +1.<br> 
The rest of the circuit computed the final value, which is -12. <br>
During the backward pass in which the chain rule is applied recursively backwards through the circuit, the add gate (which is an input to the multiply gate) learns that the gradient for its output was -4.<br> 
If we anthropomorphize the circuit as wanting to output a higher value (which can help with intuition), then we can think of the circuit as "wanting" the output of the add gate to be lower (due to negative sign), and with a *force* of 4.<br> 
To continue the recurrence and to chain the gradient, the add gate takes that gradient and multiplies it to all of the local gradients for its inputs (making the gradient on both **x** and **y** 1 * -4 = -4). <br>
Notice that this has the desired effect: If **x,y** were to decrease (responding to their negative gradient) then the add gate's output would decrease, which in turn makes the multiply gate's output increase.<br>

- 예제를 다시 참조하여 이 작업이 어떻게 작동하는지 직관해 보겠습니다.
- 추가 게이트는 입력  [-2, 5] 과 계산 출력 3을 수신했습니다.
- 게이트가 추가 연산을 계산하기 때문에 두 입력 모두에 대한 로컬 기울기는 +1입니다.
- 회로의 나머지 부분은 최종 값인 -12를 계산했습니다. 
- 체인 규칙이 회로를 통해 재귀적으로 역방향으로 적용되는 역방향 패스 동안, 추가 게이트(다중 게이트에 대한 입력)는 출력에 대한 기울기가 -4였음을 학습합니다. 
- 회로를 더 높은 값(직관에 도움이 될 수 있음)을 출력하기를 원하는 것으로 의인화하면 회로가 4의 힘으로 추가 게이트의 출력을 더 낮게 "wanting" 하는 것으로 생각할 수 있습니다. 
- 반복을 계속하고 그라데이션을 연결하기 위해 추가 게이트는 해당 그라데이션을 사용하여 입력에 대한 모든 로컬 그라데이션에 곱합니다(**x** 및 **y** 1 모두에 그라데이션을 \* -4 = -4로 만듭니다).
- 원하는 효과가 있습니다. x,y가 감소하면(음의 기울기에 따라) 추가 게이트의 출력이 감소하고, 이는 다시 곱셈 게이트의 출력을 증가시킵니다.

Backpropagation can thus be thought of as gates communicating to each other (through the gradient signal) whether they want their outputs to increase or decrease (and how strongly), so as to make the final output value higher.

- 따라서 역전파는 최종 출력 값을 더 높게 만들기 위해 출력이 증가하거나 감소하기를 원하는지(그리고 얼마나 강하게) 서로 통신하는 게이트로 생각할 수 있습니다(기울기 신호를 통해).

# 5. [Modularity: Sigmoid example](https://cs231n.github.io/optimization-2/#sigmoid)

The gates we introduced above are relatively arbitrary. <br>
Any kind of differentiable function can act as a gate, and we can group multiple gates into a single gate, or decompose a function into multiple gates whenever it is convenient.<br> 
Lets look at another expression that illustrates this point:<br>


- 위에서 소개한 게이트는 비교적 임의적입니다. 
- 모든 종류의 미분 가능한 함수는 게이트 역할을 할 수 있으며 편리할 때마다 여러 게이트를 단일 게이트로 그룹화하거나 함수를 여러 게이트로 분해할 수 있습니다. 
- 이 점을 설명하는 다른 표현을 살펴보겠습니다.

- 미분 가능하다면 게이트 역할을 할 수 있다.

$$
f(w,x) = \frac{1}{1+e^{-(w_0x_0 + w_1x_1 + w_2)}}
$$



as we will see later in the class, this expression describes a 2-dimensional neuron (with inputs **x** and weights **w**) that uses the *sigmoid activation* function. <br>
But for now lets think of this very simply as just a function from inputs *w,x* to a single number. <br>
The function is made up of multiple gates. In addition to the ones described already above (add, mul, max), there are four more:<br>
-   수업 후반에 보게 될 것처럼, 이 표현식은 시그모이드 활성화 함수를 사용하는 2차원 뉴런(입력 x와 가중치 w)을 설명합니다. <br>
- 하지만 지금은 이것을 단순히 입력 w,x에서 단일 숫자로의 함수로 생각해 보겠습니다.<br>
- 기능은 여러 개의 게이트로 구성되어 있습니다. 위에서 이미 설명한 것들(add, mul, max) 외에도 다음 네 가지가 더 있습니다:<br>

$$
f(x) = \frac{1}{x}
\hspace{1in} \rightarrow \hspace{1in}
\frac{df}{dx} = -1/x^2
\\\\
f_c(x) = c + x
\hspace{1in} \rightarrow \hspace{1in}
\frac{df}{dx} = 1
\\\\
f(x) = e^x
\hspace{1in} \rightarrow \hspace{1in}
\frac{df}{dx} = e^x
\\\\
f_a(x) = ax
\hspace{1in} \rightarrow \hspace{1in}
\frac{df}{dx} = a
$$

Where the functions \\(f_c, f_a\\) translate the input by a constant of \\(c\\) and scale the input by a constant of \\(a\\), respectively. <br>
These are technically special cases of addition and multiplication, but we introduce them as (new) unary gates here since we do not need the gradients for the constants \\(c,a\\).<br> 
The full circuit then looks as follows:<br>

- 여기서 함수 \\(f_c, f_a\\) 는 \\(c\\)의 상수로 입력을 변환하고 \\(a\\)의 상수로 입력을 각각 축척합니다. 이것들은 기술적으로 덧셈과 곱셈의 특별한
- 경우이지만, 우리는 상수 \\(c,a\\)에 대한 기울기가 필요하지 않기 때문에 여기서 (새로운) 단항 게이트로 소개합니다. 
- 그러면 전체 회로는 다음과 같이 표시됩니다:

<div class="fig figleft fighighlight">
<svg style="max-width: 100%" viewbox="0 0 799 306"><g transform="scale(0.8)"><defs><marker id="arrowhead" refX="6" refY="2" markerWidth="6" markerHeight="4" orient="auto"><path d="M 0,0 V 4 L6,2 Z"></path></marker></defs><line x1="50" y1="30" x2="90" y2="30" stroke="black" stroke-width="1"></line><text x="55" y="24" font-size="16" fill="green">2.00</text><text x="55" y="47" font-size="16" fill="red">-0.20</text><text x="45" y="24" font-size="16" text-anchor="end" fill="black">w0</text><line x1="50" y1="100" x2="90" y2="100" stroke="black" stroke-width="1"></line><text x="55" y="94" font-size="16" fill="green">-1.00</text><text x="55" y="117" font-size="16" fill="red">0.39</text><text x="45" y="94" font-size="16" text-anchor="end" fill="black">x0</text><line x1="50" y1="170" x2="90" y2="170" stroke="black" stroke-width="1"></line><text x="55" y="164" font-size="16" fill="green">-3.00</text><text x="55" y="187" font-size="16" fill="red">-0.39</text><text x="45" y="164" font-size="16" text-anchor="end" fill="black">w1</text><line x1="50" y1="240" x2="90" y2="240" stroke="black" stroke-width="1"></line><text x="55" y="234" font-size="16" fill="green">-2.00</text><text x="55" y="257" font-size="16" fill="red">-0.59</text><text x="45" y="234" font-size="16" text-anchor="end" fill="black">x1</text><line x1="50" y1="310" x2="90" y2="310" stroke="black" stroke-width="1"></line><text x="55" y="304" font-size="16" fill="green">-3.00</text><text x="55" y="327" font-size="16" fill="red">0.20</text><text x="45" y="304" font-size="16" text-anchor="end" fill="black">w2</text><line x1="170" y1="65" x2="210" y2="65" stroke="black" stroke-width="1"></line><text x="175" y="59" font-size="16" fill="green">-2.00</text><text x="175" y="82" font-size="16" fill="red">0.20</text><circle cx="130" cy="65" fill="white" stroke="black" stroke-width="1" r="20"></circle><text x="130" y="75" font-size="20" fill="black" text-anchor="middle">*</text><line x1="90" y1="30" x2="110" y2="65" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="90" y1="100" x2="110" y2="65" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="150" y1="65" x2="170" y2="65" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="170" y1="205" x2="210" y2="205" stroke="black" stroke-width="1"></line><text x="175" y="199" font-size="16" fill="green">6.00</text><text x="175" y="222" font-size="16" fill="red">0.20</text><circle cx="130" cy="205" fill="white" stroke="black" stroke-width="1" r="20"></circle><text x="130" y="215" font-size="20" fill="black" text-anchor="middle">*</text><line x1="90" y1="170" x2="110" y2="205" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="90" y1="240" x2="110" y2="205" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="150" y1="205" x2="170" y2="205" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="290" y1="135" x2="330" y2="135" stroke="black" stroke-width="1"></line><text x="295" y="129" font-size="16" fill="green">4.00</text><text x="295" y="152" font-size="16" fill="red">0.20</text><circle cx="250" cy="135" fill="white" stroke="black" stroke-width="1" r="20"></circle><text x="250" y="140" font-size="20" fill="black" text-anchor="middle">+</text><line x1="210" y1="65" x2="230" y2="135" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="210" y1="205" x2="230" y2="135" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="270" y1="135" x2="290" y2="135" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="410" y1="222" x2="450" y2="222" stroke="black" stroke-width="1"></line><text x="415" y="216" font-size="16" fill="green">1.00</text><text x="415" y="239" font-size="16" fill="red">0.20</text><circle cx="370" cy="222" fill="white" stroke="black" stroke-width="1" r="20"></circle><text x="370" y="227" font-size="20" fill="black" text-anchor="middle">+</text><line x1="330" y1="135" x2="350" y2="222" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="90" y1="310" x2="350" y2="222" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="390" y1="222" x2="410" y2="222" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="530" y1="222" x2="570" y2="222" stroke="black" stroke-width="1"></line><text x="535" y="216" font-size="16" fill="green">-1.00</text><text x="535" y="239" font-size="16" fill="red">-0.20</text><circle cx="490" cy="222" fill="white" stroke="black" stroke-width="1" r="20"></circle><text x="490" y="227" font-size="20" fill="black" text-anchor="middle">*-1</text><line x1="450" y1="222" x2="470" y2="222" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="510" y1="222" x2="530" y2="222" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="650" y1="222" x2="690" y2="222" stroke="black" stroke-width="1"></line><text x="655" y="216" font-size="16" fill="green">0.37</text><text x="655" y="239" font-size="16" fill="red">-0.53</text><circle cx="610" cy="222" fill="white" stroke="black" stroke-width="1" r="20"></circle><text x="610" y="227" font-size="20" fill="black" text-anchor="middle">exp</text><line x1="570" y1="222" x2="590" y2="222" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="630" y1="222" x2="650" y2="222" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="770" y1="222" x2="810" y2="222" stroke="black" stroke-width="1"></line><text x="775" y="216" font-size="16" fill="green">1.37</text><text x="775" y="239" font-size="16" fill="red">-0.53</text><circle cx="730" cy="222" fill="white" stroke="black" stroke-width="1" r="20"></circle><text x="730" y="227" font-size="20" fill="black" text-anchor="middle">+1</text><line x1="690" y1="222" x2="710" y2="222" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="750" y1="222" x2="770" y2="222" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="890" y1="222" x2="930" y2="222" stroke="black" stroke-width="1"></line><text x="895" y="216" font-size="16" fill="green">0.73</text><text x="895" y="239" font-size="16" fill="red">1.00</text><circle cx="850" cy="222" fill="white" stroke="black" stroke-width="1" r="20"></circle><text x="850" y="227" font-size="20" fill="black" text-anchor="middle">1/x</text><line x1="810" y1="222" x2="830" y2="222" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="870" y1="222" x2="890" y2="222" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line></g></svg>
<div style="clear:both;"></div>
</div>

Example circuit for a 2D neuron with a sigmoid activation function. <br>
The inputs are [x0,x1] and the (learnable) weights of the neuron are [w0,w1,w2].<br>
As we will see later, the neuron computes a dot product with the input and then its activation is softly squashed by the sigmoid function to be in range from 0 to 1.<br>

- 시그모이드 활성화 기능을 갖는 2D 뉴런에 대한 예시 회로. 
- 입력은 [x0,x1] 이고 뉴런의 (학습 가능한) 가중치는 [w0,w1,w2]입니다.
- 우리가 나중에 보게 될 것처럼, 뉴런은 입력으로 내적을 계산하고 그 활성화는 0 \~ 1 사이에 있는 sigmoid 함수 결과에 의해 값을 가질 것이다.

In the example above, we see a long chain of function applications that operates on the result of the dot product between **w,x**. <br>
The function that these operations implement is called the *sigmoid function* \\(\sigma(x)\\). <br>
It turns out that the derivative of the sigmoid function with respect to its input simplifies if you perform the derivation (after a fun tricky part where we add and subtract a 1 in the numerator):<br>

- 위의 예제에서 **w,x**사이의 내적 결과에서 작동하는 긴 함수 응용 프로그램 체인을 볼 수 있습니다. 
- 이러한 연산이 구현하는 함수를 시그모이드 함수 \\(\sigma(x)\\)라고 합니다. 
- (분자에서 1을 더하고 빼는 재미있는 까다로운 부분 이후에) 입력에 대한 시그모이드 함수의 도함수가 단순해진다는 것이 밝혀졌습니다:

$$
\sigma(x) = \frac{1}{1+e^{-x}} \\\\
\rightarrow \hspace{0.3in} \frac{d\sigma(x)}{dx} = \frac{e^{-x}}{(1+e^{-x})^2} = \left( \frac{1 + e^{-x} - 1}{1 + e^{-x}} \right) \left( \frac{1}{1+e^{-x}} \right)
= \left( 1 - \sigma(x) \right) \sigma(x)
$$

As we see, the gradient turns out to simplify and becomes surprisingly simple. <br>
For example, the sigmoid expression receives the input 1.0 and computes the output 0.73 during the forward pass.<br> 
The derivation above shows that the *local* gradient would simply be (1 - 0.73) * 0.73 ~= 0.2, as the circuit computed before (see the image above), except this way it would be done with a single, simple and efficient expression (and with less numerical issues).<br> 
Therefore, in any real practical application it would be very useful to group these operations into a single gate. Lets see the backprop for this neuron in code:<br>

- 보시다시피, 기울기는 단순화되고 놀라울 정도로 단순화됩니다. 
- 예를 들어, Sigmoid 식은 입력 1.0을 수신하고 정방향 통과 중에 출력 0.73을 계산합니다.
- 위의 기울기는 로컬 기울기가 이전에 계산된 회로(위의 아미지 참조)와 같이 (1 - 0.73) \* 0.73 \~ = 0.2임을 보여줍니다. 단,이러한 방식은 단일하고 단순하며 효율적인 식을 사용하여 수행됩니다(수치 문제가 더 적음). 
- 따라서 실제 응용 분야에서는 이러한 작업을 단일 게이트로 그룹화하는 것이 매우 유용합니다. 코드에서 이 뉴런에 대한 backprop을 보겠습니다:

```python
w = [2,-3,-3] # assume some random weights and data
x = [-1, -2]

# forward pass
dot = w[0]*x[0] + w[1]*x[1] + w[2]
f = 1.0 / (1 + math.exp(-dot)) # sigmoid function

# backward pass through the neuron (backpropagation)
ddot = (1 - f) * f # gradient on dot variable, using the sigmoid gradient derivation
dx = [w[0] * ddot, w[1] * ddot] # backprop into x
dw = [x[0] * ddot, x[1] * ddot, 1.0 * ddot] # backprop into w
# we're done! we have the gradients on the inputs to the circuit
```

**Implementation protip: staged backpropagation**. <br>
As shown in the code above, in practice it is always helpful to break down the forward pass into stages that are easily backpropped through.<br>
For example here we created an intermediate variable `dot` which holds the output of the dot product between `w` and `x`. <br>
During backward pass we then successively compute (in reverse order) the corresponding variables (e.g. `ddot`, and ultimately `dw, dx`) that hold the gradients of those variables.<br>

- 구현 팁: 단계적 역전파. 
- 위의 코드에서 볼 수 있듯이 실제로는 순방향 패스를 쉽게 역전파할 수 있는 단계로 나누는 것이 항상 도움이 됩니다. 
- 예를 들어 여기서 우리는 `w`와 `x` 사이의 내적의 출력을 보유하는 중간 변수 `dot`를 만들었습니다.
- 역방향 전달 중에 해당 변수의 기울기를 유지하는 해당 변수(예: `ddot` 및 궁극적으로 `dw, dx`)를 연속적으로(역순으로) 계산합니다.

- 순방향 패스를 쉽게 역전파할 수 있는 단계로 나누는 것이 항상 좋다. 기울기를 유지하는 해당 변수를 역순으로 계산한다.

The point of this section is that the details of how the backpropagation is performed, and which parts of the forward function we think of as gates, is a matter of convenience. 
It helps to be aware of which parts of the expression have easy local gradients, so that they can be chained together with the least amount of code and effort.

- 이 섹션의 요점은 역전파가 수행되는 방법에 대한 세부 사항과 우리가 게이트로 생각하는 순방향 기능 부분이 편의상 문제라는 것입니다.
- 표현식의 어느 부분에 쉬운 로컬 그래디언트가 있는지 인식하는 데 도움이 되므로 최소한의 코드와 노력으로 함께 연결할 수 있습니다.

# 6. [Backprop in practice: Staged computation](https://cs231n.github.io/optimization-2/#staged)

Lets see this with another example. Suppose that we have a function of the form:

-   다른 예를 들어 보겠습니다. 다음 형식의 함수가 있다고 가정합니다.

$$
f(x,y) = \frac{x + \sigma(y)}{\sigma(x) + (x+y)^2}
$$

To be clear, this function is completely useless and it's not clear why you would ever want to compute its gradient, except for the fact that it is a good example of backpropagation in practice. <br>
It is very important to stress that if you were to launch into performing the differentiation with respect to either \\(x\\) or \\(y\\), you would end up with very large and complex expressions. <br>
However, it turns out that doing so is completely unnecessary because we don't need to have an explicit function written down that evaluates the gradient. <br>
We only have to know how to compute it. Here is how we would structure the forward pass of such expression:<br>

- 확실히 하기 위해 이 함수는 완전히 쓸모가 없으며 실제로 역전파의 좋은 예라는 사실을 제외하고는 기울기를 계산하려는 이유가 명확하지 않습니다.
- x 또는 y에 대한 미분을 수행하는 경우 매우 크고 복잡한 표현식이 생성된다는 점을 강조하는 것이 매우 중요합니다. 
- 그러나 그래디언트를 평가하는 명시적인 함수를 작성할 필요가 없기 때문에 그렇게 하는 것이 완전히 불필요하다는 것이 밝혀졌습니다. 
- 우리는 그것을 계산하는 방법만 알면 됩니다. 다음은 이러한 표현의 정방향 전달을 구성하는 방법입니다.

```python
x = 3 # example values
y = -4

# forward pass
sigy = 1.0 / (1 + math.exp(-y)) # sigmoid in numerator   #(1)
num = x + sigy # numerator                               #(2)
sigx = 1.0 / (1 + math.exp(-x)) # sigmoid in denominator #(3)
xpy = x + y                                              #(4)
xpysqr = xpy**2                                          #(5)
den = sigx + xpysqr # denominator                        #(6)
invden = 1.0 / den                                       #(7)
f = num * invden # done!                                 #(8)
```

Phew, by the end of the expression we have computed the forward pass. 
Notice that we have structured the code in such way that it contains multiple intermediate variables, each of which are only simple expressions for which we already know the local gradients. 
Therefore, computing the backprop pass is easy: We'll go backwards and for every variable along the way in the forward pass (`sigy, num, sigx, xpy, xpysqr, den, invden`) we will have the same variable, but one that begins with a `d`, which will hold the gradient of the output of the circuit with respect to that variable. 
Additionally, note that every single piece in our backprop will involve computing the local gradient of that expression, and chaining it with the gradient on that expression with a multiplication. For each row, we also highlight which part of the forward pass it refers to:

- 휴, 표현식이 끝날 때까지 순방향 패스를 계산했습니다. 
- 우리는 여러 중간 변수를 포함하는 방식으로 코드를 구성했으며 각 변수는 이미 로컬 기울기를 알고 있는 단순한 표현식일 뿐입니다. 
- 따라서 역전파 패스를 계산하는 것은 쉽습니다. 역전파 패스의 모든 변수(`sigy, num, sigx, xpy, xpysqr, den, invden`)에 대해 동일한 변수를 갖지만 해당 변수에 대한 회로 출력의 기울기를 유지하는 d로 시작합니다. 
- 또한 역전파의 모든 단일 부분에는 해당 표현식의 로컬 그래디언트를 계산하고 곱셈을 통해 해당 표현식의 그래디언트와 연결하는 작업이 포함됩니다. 각 행에 대해 참조하는 정방향 전달 부분도 강조 표시합니다.

```python
# backprop f = num * invden
dnum = invden # gradient on numerator                             #(8)
dinvden = num                                                     #(8)
# backprop invden = 1.0 / den
dden = (-1.0 / (den**2)) * dinvden                                #(7)
# backprop den = sigx + xpysqr
dsigx = (1) * dden                                                #(6)
dxpysqr = (1) * dden                                              #(6)
# backprop xpysqr = xpy**2
dxpy = (2 * xpy) * dxpysqr                                        #(5)
# backprop xpy = x + y
dx = (1) * dxpy                                                   #(4)
dy = (1) * dxpy                                                   #(4)
# backprop sigx = 1.0 / (1 + math.exp(-x))
dx += ((1 - sigx) * sigx) * dsigx # Notice += !! See notes below  #(3)
# backprop num = x + sigy
dx += (1) * dnum                                                  #(2)
dsigy = (1) * dnum                                                #(2)
# backprop sigy = 1.0 / (1 + math.exp(-y))
dy += ((1 - sigy) * sigy) * dsigy                                 #(1)
# done! phew
```

Notice a few things:<br>
- 주의사항

**Cache forward pass variables**. <br>
To compute the backward pass it is very helpful to have some of the variables that were used in the forward pass.<br>
In practice you want to structure your code so that you cache these variables, and so that they are available during backpropagation.<br>
If this is too difficult, it is possible (but wasteful) to recompute them.<br>

- 캐시 전달 전달 변수.
- 역방향 패스를 계산하려면 순방향 패스에서 사용된 일부 변수를 갖는 것이 매우 유용합니다. 
- 실제로 이러한 변수를 캐시하고 역전파 중에 사용할 수 있도록 코드를 구조화하려고 합니다. 
- 이것이 너무 어려우면 다시 계산하는 것이 가능하지만 낭비입니다.

**Gradients add up at forks**. <br>
The forward expression involves the variables **x,y** multiple times, so when we perform backpropagation we must be careful to use `+=` instead of `=` to accumulate the gradient on these variables (otherwise we would overwrite it). 
This follows the *multivariable chain rule* in Calculus, which states that if a variable branches out to different parts of the circuit, then the gradients that flow back to it will add.

- 그라디언트는 포크에서 합산됩니다.<br> 
- 순방향 표현식은 변수 **x,y** 를 여러 번 포함하므로 역전파를 수행할 때 `=` 대신 `+=`를 사용하여 이러한 변수에 그래디언트를 누적하도록 주의해야 합니다(그렇지 않으면 덮어씁니다).<br>
- 이것은 변수가 회로의 다른 부분으로 분기되면 변수로 다시 흐르는 그래디언트가 추가된다는 미적분학의 다중 변수 체인 규칙을 따릅니다.<br>

# 7. [Patterns in backward flow](https://cs231n.github.io/optimization-2/#patterns)

It is interesting to note that in many cases the backward-flowing gradient can be interpreted on an intuitive level. <br>
For example, the three most commonly used gates in neural networks (*add,mul,max*), all have very simple interpretations in terms of how they act during backpropagation.<br> 
Consider this example circuit:<br>


- 많은 경우에 역방향 기울기가 직관적인 수준에서 해석될 수 있다는 점은 흥미롭습니다.
- 예를 들어, 신경망에서 가장 일반적으로 사용되는 세 가지 게이트(add,mul,max)는 모두 역전파 중에 작동하는 방식에 대해 매우 간단한 해석을 가집니다.
- 이 예제 회로를 고려하십시오.

<div class="fig figleft fighighlight">
<svg style="max-width: 80%" viewbox="0 0 460 290"><g transform="scale(1)"><defs><marker id="arrowhead" refX="6" refY="2" markerWidth="6" markerHeight="4" orient="auto"><path d="M 0,0 V 4 L6,2 Z"></path></marker></defs><line x1="50" y1="30" x2="90" y2="30" stroke="black" stroke-width="1"></line><text x="55" y="24" font-size="16" fill="green">3.00</text><text x="55" y="47" font-size="16" fill="red">-8.00</text><text x="45" y="24" font-size="16" text-anchor="end" fill="black">x</text><line x1="50" y1="100" x2="90" y2="100" stroke="black" stroke-width="1"></line><text x="55" y="94" font-size="16" fill="green">-4.00</text><text x="55" y="117" font-size="16" fill="red">6.00</text><text x="45" y="94" font-size="16" text-anchor="end" fill="black">y</text><line x1="50" y1="170" x2="90" y2="170" stroke="black" stroke-width="1"></line><text x="55" y="164" font-size="16" fill="green">2.00</text><text x="55" y="187" font-size="16" fill="red">2.00</text><text x="45" y="164" font-size="16" text-anchor="end" fill="black">z</text><line x1="50" y1="240" x2="90" y2="240" stroke="black" stroke-width="1"></line><text x="55" y="234" font-size="16" fill="green">-1.00</text><text x="55" y="257" font-size="16" fill="red">0.00</text><text x="45" y="234" font-size="16" text-anchor="end" fill="black">w</text><line x1="170" y1="65" x2="210" y2="65" stroke="black" stroke-width="1"></line><text x="175" y="59" font-size="16" fill="green">-12.00</text><text x="175" y="82" font-size="16" fill="red">2.00</text><circle cx="130" cy="65" fill="white" stroke="black" stroke-width="1" r="20"></circle><text x="130" y="75" font-size="20" fill="black" text-anchor="middle">*</text><line x1="90" y1="30" x2="110" y2="65" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="90" y1="100" x2="110" y2="65" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="150" y1="65" x2="170" y2="65" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="170" y1="205" x2="210" y2="205" stroke="black" stroke-width="1"></line><text x="175" y="199" font-size="16" fill="green">2.00</text><text x="175" y="222" font-size="16" fill="red">2.00</text><circle cx="130" cy="205" fill="white" stroke="black" stroke-width="1" r="20"></circle><text x="130" y="210" font-size="20" fill="black" text-anchor="middle">max</text><line x1="90" y1="170" x2="110" y2="205" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="90" y1="240" x2="110" y2="205" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="150" y1="205" x2="170" y2="205" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="290" y1="135" x2="330" y2="135" stroke="black" stroke-width="1"></line><text x="295" y="129" font-size="16" fill="green">-10.00</text><text x="295" y="152" font-size="16" fill="red">2.00</text><circle cx="250" cy="135" fill="white" stroke="black" stroke-width="1" r="20"></circle><text x="250" y="140" font-size="20" fill="black" text-anchor="middle">+</text><line x1="210" y1="65" x2="230" y2="135" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="210" y1="205" x2="230" y2="135" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="270" y1="135" x2="290" y2="135" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="410" y1="135" x2="450" y2="135" stroke="black" stroke-width="1"></line><text x="415" y="129" font-size="16" fill="green">-20.00</text><text x="415" y="152" font-size="16" fill="red">1.00</text><circle cx="370" cy="135" fill="white" stroke="black" stroke-width="1" r="20"></circle><text x="370" y="140" font-size="20" fill="black" text-anchor="middle">*2</text><line x1="330" y1="135" x2="350" y2="135" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="390" y1="135" x2="410" y2="135" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line></g></svg>
<div style="clear:both;"></div>
</div>

An example circuit demonstrating the intuition behind the operations that backpropagation performs during the backward pass in order to compute the gradients on the inputs.<br> 
Sum operation distributes gradients equally to all its inputs. Max operation routes the gradient to the higher input. <br>
Multiply gate takes the input activations, swaps them and multiplies by its gradient.<br>

- 입력에 대한 그래디언트를 계산하기 위해 역전파가 역방향 패스 중에 수행하는 작업 뒤에 있는 직관을 보여주는 예제 회로입니다.
- 합계 연산은 기울기를 모든 입력에 균등하게 분배합니다. Max 연산은 그래디언트를 더 높은 입력으로 라우팅합니다. 
- Multiply gate는 입력 활성화를 가져와 이를 교환하고 기울기를 곱합니다.

Looking at the diagram above as an example, we can see that:
- 예를 들어 위의 다이어그램을 보면 다음을 볼 수 있습니다.

The **add gate** always takes the gradient on its output and distributes it equally to all of its inputs, regardless of what their values were during the forward pass.<br> 
This follows from the fact that the local gradient for the add operation is simply +1.0, <br>
so the gradients on all inputs will exactly equal the gradients on the output because it will be multiplied by x1.0 (and remain unchanged).<br> 
In the example circuit above, note that the + gate routed the gradient of 2.00 to both of its inputs, equally and unchanged.<br>

- 더하기 게이트는 항상 출력에서 그래디언트를 가져오고 포워드 패스 동안의 값에 관계없이 모든 입력에 균등하게 분배합니다.
- 이것은 더하기 작업에 대한 로컬 그래디언트가 단순히 +1.0이라는 사실에서 비롯됩니다.
- 따라서 모든 입력의 그래디언트는 x1.0을 곱하고 변경되지 않은 상태로 유지되기 때문에 출력의 그래디언트와 정확히 동일합니다. 
- 위의 예제 회로에서 + 게이트는 기울기 2.00을 두 입력 모두에 동일하게 변경하지 않고 라우팅했습니다.

The **max gate** routes the gradient.<br>
Unlike the add gate which distributed the gradient unchanged to all its inputs, the max gate distributes the gradient (unchanged) to exactly one of its inputs (the input that had the highest value during the forward pass).<br> 
This is because the local gradient for a max gate is 1.0 for the highest value, and 0.0 for all other values. <br>
In the example circuit above, the max operation routed the gradient of 2.00 to the **z** variable, which had a higher value than **w**, and the gradient on **w** remains zero.<br>

- 최대 게이트는 그래디언트를 라우팅합니다. 
- 기울기를 변경하지 않고 모든 입력에 배포하는 추가 게이트와 달리 최대 게이트는 기울기(변경되지 않은)를 정확히 입력 중 하나(정방향 패스 중에 가장 높은 값을 가진 입력)에 배포합니다. 
- 이는 최대 게이트의 로컬 그래디언트가 가장 높은 값의 경우 1.0이고 다른 모든 값의 경우 0.0이기 때문입니다. 
- 위의 예제 회로에서 max 연산은 2.00의 기울기를 w보다 높은 값을 가진 z 변수로 라우팅했으며 w의 기울기는 0으로 유지됩니다.

The **multiply gate** is a little less easy to interpret. Its local gradients are the input values (except switched), and this is multiplied by the gradient on its output during the chain rule.<br>
In the example above, the gradient on **x** is -8.00, which is -4.00 x 2.00.<br>

- 곱하기 게이트는 해석하기가 약간 덜 쉽습니다. 로컬 그래디언트는 입력 값(스위치 제외)이며 체인 규칙 동안 출력에 그래디언트를 곱합니다. 
- 위의 예에서 x의 그래디언트는 -8.00이며 이는 -4.00 x 2.00입니다.

*Unintuitive effects and their consequences*.<br> 
Notice that if one of the inputs to the multiply gate is very small and the other is very big, then the multiply gate will do something slightly unintuitive:<br> 
it will assign a relatively huge gradient to the small input and a tiny gradient to the large input. <br>
Note that in linear classifiers where the weights are dot producted \\(w^Tx_i\\) (multiplied) with the inputs, this implies that the scale of the data has an effect on the magnitude of the gradient for the weights.<br> 
For example, if you multiplied all input data examples \\(x_i\\) by 1000 during preprocessing, then the gradient on the weights will be 1000 times larger, and you'd have to lower the learning rate by that factor to compensate.<br>
This is why preprocessing matters a lot, sometimes in subtle ways! And having intuitive understanding for how the gradients flow can help you debug some of these cases.<br>

- 직관적이지 않은 효과와 그 결과. 
- 곱셈 게이트에 대한 입력 중 하나가 매우 작고 다른 하나가 매우 큰 경우 곱셈 게이트는 약간 직관적이지 않은 작업을 수행합니다. 
- 즉, 작은 입력에는 상대적으로 큰 기울기를 할당하고 큰 입력에는 작은 기울기를 할당합니다. 
- 가중치가 입력과 내적 \\(w^Tx_i\\)(곱해짐)되는 선형 분류기에서 이는 데이터의 척도가 가중치의 그래디언트 크기에 영향을 미친다는 것을 의미합니다. 
- 예를 들어 전처리 중에 모든 입력 데이터 예제 \\(x_i\\)에 1000을 곱하면 가중치의 그래디언트가 1000배 더 커지고 보상을 위해 학습률을 해당 계수만큼 낮춰야 합니다.
- 이것이 전처리가 때로는 미묘하게 중요한 이유입니다! 그래디언트 흐름을 직관적으로 이해하면 이러한 사례 중 일부를  디버깅하는 데 도움이 될 수 있습니다.

# 8. [Gradients for vectorized operations](https://cs231n.github.io/optimization-2/#mat)

The above sections were concerned with single variables, but all concepts extend in a straight-forward manner to matrix and vector operations. <br>
However, one must pay closer attention to dimensions and transpose operations.<br>

- 위의 섹션은 단일 변수와 관련이 있지만 모든 개념은 행렬 및 벡터 연산으로 직진 방식으로 확장됩니다.
- 그러나 치수 및 전치 작업에 더 많은 주의를 기울여야 합니다.

**Matrix-Matrix multiply gradient**. Possibly the most tricky operation is the matrix-matrix multiplication (which generalizes all matrix-vector and vector-vector) multiply operations:

- 매트릭스-매트릭스 곱하기 그래디언트. 아마도 가장 까다로운 연산은 행렬-행렬 곱셈(모든 행렬-벡터 및 벡터-벡터를 일반화함) 곱셈 연산일 것입니다.

```python
# forward pass
W = np.random.randn(5, 10)
X = np.random.randn(10, 3)
D = W.dot(X)

# now suppose we had the gradient on D from above in the circuit
dD = np.random.randn(*D.shape) # same shape as D
dW = dD.dot(X.T) #.T gives the transpose of the matrix
dX = W.T.dot(dD)
```

*Tip: use dimension analysis!* <br>
Note that you do not need to remember the expressions for `dW` and `dX`  because they are easy to re-derive based on dimensions. <br> 
For instance, we know that the gradient on the weights `dW` must be of the same size as `W` after it is computed, and that it must depend on matrix multiplication of `X` and `dD` (as is the case when both `X,W` are single numbers and not matrices).<br> 
There is always exactly one way of achieving this so that the dimensions work out. <br>
For example, `X` is of size [10 x 3] and `dD` of size [5 x 3], so if we want `dW` and `W` has shape [5 x 10], then the only way of achieving this is with `dD.dot(X.T)`, as shown above.<br>

- 팁: 차원 분석을 사용하십시오! 
- `dW` 및 `dX` 치수를 기준으로 쉽게 다시 도출할 수 있으므로 식을 기억할 필요가 없습니다. 
- 예를 들어 가중치 `dW`의 기울기는 계산된 후 `W`와 같은 크기여야 하고 `X`와 `dD`의 행렬 곱셈에 의존해야 한다는 것을 알고 있습니다(X,W가 둘 다 단일 숫자인 경우의 경우) 행렬이 아님).
- 치수가 제대로 작동하도록 이를 달성하는 방법은 항상 정확히 한 가지입니다.
- 예를 들어 `X`의 크기는 [10 x 3]이고 `dD`의 크기는 [5 x 3]이므로 `dW`와 `W`의 모양이 [5 x 10]이면 이를 달성하는 유일한 방법은 `dD.dot(X.T)`, 위와 같이 표시됩니다.

**Work with small, explicit examples**. 
Some people may find it difficult at first to derive the gradient updates for some vectorized expressions.
Our recommendation is to explicitly write out a minimal vectorized example, derive the gradient on paper and then generalize the pattern to its efficient, vectorized form.

- 작고 명시적인 예를 사용하여 작업합니다. 
- 어떤 사람들은 일부 벡터화된 표현에 대한 그래디언트 업데이트를 도출하는 것이 처음에는 어려울 수 있습니다. 
- 최소한의 벡터화된 예제를 명시적으로 작성하고 종이에 그래디언트를 도출한 다음 패턴을 효율적이고 벡터화된 형태로 일반화하는 것이 좋습니다.

Erik Learned-Miller has also written up a longer related document on taking matrix/vector derivatives which you might find helpful.<br> 
[Find it here](http://cs231n.stanford.edu/vecDerivs.pdf).<br>


- Erik Learned-Miller는 도움이 될 수 있는 행렬/벡터 도함수를 사용하는 방법에 대한 더 긴 관련 문서도 작성했습니다. 
- 여기에서 찾으십시오.[Find it here](http://cs231n.stanford.edu/vecDerivs.pdf)

# 9. [Summary](https://cs231n.github.io/optimization-2/#summary)


We developed intuition for what the gradients mean, how they flow backwards in the circuit, and how they communicate which part of the circuit should increase or decrease and with what force to make the final output higher.

- 우리는 기울기가 무엇을 의미하는지, 기울기가 회로에서 어떻게 거꾸로 흐르는지, 회로의 어떤 부분이 증가하거나 감소해야 하는지, 그리고 최종 출력을 더 높게 만들기 위해 어떤 힘으로 의사소통하는지에 대한 직관을 개발했습니다.

We discussed the importance of **staged computation** for practical implementations of backpropagation.<br>
You always want to break up your function into modules for which you can easily derive local gradients, and then chain them with chain rule.<br>
Crucially, you almost never want to write out these expressions on paper and differentiate them symbolically in full, because you never need an explicit mathematical equation for the gradient of the input variables.<br> 
Hence, decompose your expressions into stages such that you can differentiate every stage independently (the stages will be matrix vector multiplies, or max operations, or sum operations, etc.) and then backprop through the variables one step at a time.<br>

- 역전파의 실제 구현을 위한 단계적 계산의 중요성에 대해 논의했습니다. 
- 항상 함수를 로컬 그래디언트를 쉽게 파생할 수 있는 모듈로 분할한 다음 체인 규칙으로 연결하려고 합니다. 
- 결정적으로 입력 변수의 그래디언트에 대해 명시적인 수학 방정식이 필요하지 않기 때문에 이러한 식을 종이에 작성하고 완전히 기호로 미분하는 것을 거의 원하지 않습니다. 
- 따라서 모든 단계를 독립적으로 구별할 수 있도록 표현식을 단계로 분해한 다음(단계는 행렬 벡터 곱, 최대 연산 또는 합계 연산 등이 됨) 한 번에 한 단계씩 변수를 역전파합니다.

In the next section we will start to define neural networks, and backpropagation will allow us to efficiently compute the gradient of a loss function with respect to its parameters. <br>
In other words, we're now ready to train neural nets, and the most conceptually difficult part of this class is behind us! ConvNets will then be a small step away.y. <br>

-   다음 섹션에서는 신경망을 정의하기 시작하고 역전파를 통해 매개변수와 관련하여 손실 함수의 그래디언트를 효율적으로 계산할 수 있습니다. 
- 즉, 이제 우리는 신경망을 훈련시킬 준비가 되었으며, 이 수업에서 가장 개념적으로 어려운 부분은 뒤에 있습니다! ConvNets는 작은 발걸음이 될 것입니다.


# 10. Additional references
- https://cs231n.github.io
