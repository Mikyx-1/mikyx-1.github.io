---
title: 'Bits, Range, and Precision: How Numerical Data Types Actually Work'
date: 2026-08-27
tags:
  - Numerical Computing
  - Machine Learning
  - Quantization
---

When we say that a model is "in FP16" or that a tensor is "INT8", we compress several decisions into one label. Which bit patterns represent values? Which real numbers are missing? How are intermediate results computed? In what format are sums accumulated? What happens when a result lies between two representable numbers?

Those questions are not implementation trivia. They determine whether a training run overflows, whether adding a small gradient changes a parameter, how much memory a model occupies, and what information is lost before a quantization algorithm even begins.

This article develops the common numerical data types used in machine learning from their bits. We will derive their ranges, encode and decode values by hand, examine how arithmetic is rounded, and establish useful error bounds. Statements about mathematical formats are based on IEEE 754-2019 [1](#ref-1); statements about particular accelerator behavior are identified separately because a storage format alone does not determine a complete computation.

## 1. A dtype is more than a number of bits

A numerical data type defines at least two things:

1. a finite set of representable values and an encoding from bit patterns to those values;
2. rules for operations, conversions, rounding, and exceptional values.

In machine-learning systems, it is helpful to separate three roles that are often hidden behind one dtype label:

- **Storage dtype:** how weights, activations, or optimizer states occupy memory.
- **Operand or compute dtype:** the precision presented to an arithmetic unit.
- **Accumulator dtype:** the precision used to collect products or partial sums.

For example, a matrix multiplication can read BF16 operands, multiply them at BF16 precision, accumulate products in FP32, and finally store a BF16 output. Google describes this BF16-multiply/FP32-accumulate pattern for its TPUs [4](#ref-4), while PyTorch documents several backend-dependent choices for reduced-precision matrix multiplication [7](#ref-7). Therefore, "BF16 matrix multiplication" is not a complete numerical specification by itself.

This distinction will become essential in Part 2. A four-bit weight may occupy four bits in a checkpoint but be unpacked and dequantized before arithmetic. Storage compression and low-precision computation are related, but they are not identical.

## 2. Integers: exact values inside a finite range

### 2.1 Unsigned binary

For an $n$-bit unsigned integer with bits $b_{n-1}\ldots b_1b_0$, positional notation gives

$$
x = \sum_{i=0}^{n-1} b_i 2^i, \qquad b_i \in \{0,1\}.
$$

The smallest value is obtained when every bit is zero. The largest is obtained when every bit is one:

$$
\sum_{i=0}^{n-1}2^i = 2^n-1.
$$

The equality follows from the finite geometric-series formula. Thus an unsigned $n$-bit integer represents exactly the integers in

$$
0 \leq x \leq 2^n-1.
$$

UINT8, for example, represents the 256 integers from 0 through 255. It does not represent $12.5$, and it does not approximate it automatically; an application must supply a scale or a conversion rule.

### 2.2 Two's-complement signed integers

Most contemporary systems encode signed integers using two's complement. Its value equation is

$$
x = -b_{n-1}2^{n-1} + \sum_{i=0}^{n-2} b_i2^i.
$$

The most significant bit has negative weight. Setting only that bit gives $-2^{n-1}$; clearing it and setting all remaining bits gives $2^{n-1}-1$. Therefore

$$
-2^{n-1} \leq x \leq 2^{n-1}-1.
$$

For INT8 the range is $[-128,127]$. Decode the bit pattern `11110110` directly:

$$
-128 + 64 + 32 + 16 + 4 + 2 = -10.
$$

The familiar "invert the bits and add one" procedure gives the magnitude of a negative two's-complement number, but the weighted sum above is the definition and works without a special case.

Integer addition and multiplication are exact only when the mathematical result remains in range. At the machine-instruction level, fixed-width arithmetic may wrap modulo $2^n$; numerical libraries may instead widen the accumulator or detect overflow. Quantizers usually **clip** before storing. These are different policies, so one should never infer overflow behavior solely from the word `INT8`.

### 2.3 INT4 and packing

A signed four-bit two's-complement value has range $[-8,7]$. General-purpose machines do not usually address individual four-bit words. Libraries commonly pack two INT4 values into one byte, then unpack or decode them in a kernel. Consequently, a nominal four-bit tensor also needs layout information: nibble order, signedness, grouping, scales, and sometimes zero-points.

<figure style="text-align: center">
  <img src="/images/posts/numerical-dtypes/integer-encoding.svg" alt="Unsigned and two's-complement interpretations of four bits, with values zero to fifteen above and negative eight to seven below">
  <figcaption><b>Figure 1.</b> The same four bits can denote different values because an encoding is part of the dtype. Diagram by the author.</figcaption>
</figure>

## 3. Fixed point: the bridge to integer quantization

An integer code can stand for a real value when accompanied by a scale $s$ and, optionally, a zero-point $z$:

$$
\hat{x} = s(q-z).
$$

Here $q$ is the stored integer and $\hat{x}$ is the represented real value. With $s=0.25$, $z=0$, and $q=27$, the represented value is $6.75$. Adjacent integer codes are always separated by $0.25$, so this is a **uniform** grid.

Traditional fixed-point notation fixes the binary point by convention. For instance, a format with four fractional bits represents $q/16$. The scale-and-zero-point expression is more general and is exactly the form we will use for affine integer quantization in Part 2.

Notice what the dtype cannot tell us here. INT8 provides the code range, but not the real-value range. If $s=0.25$ and $z=0$, the INT8 codes represent values from $-32$ to $31.75$. If $s=0.01$, the same bits cover only $[-1.28,1.27]$ with finer spacing. Range and resolution trade against each other.

## 4. Floating point: moving the binary point

Fixed point uses a constant spacing. Floating point uses an exponent to make the spacing grow with magnitude. A binary floating-point encoding divides its bits into:

- one sign bit $s$;
- $w$ exponent bits storing an unsigned integer $E$;
- $t$ trailing significand bits storing an unsigned integer $F$.

The precision is $p=t+1$ significant binary digits for normal numbers because the leading digit is implicit.

<figure style="text-align: center">
  <img src="/images/posts/numerical-dtypes/float-layouts.svg" alt="Bit layouts for FP32, TF32 computation, FP16, BF16, FP8 E4M3, and FP8 E5M2">
  <figcaption><b>Figure 2.</b> Common ML floating-point layouts. TF32 is shown as a compute format, not a 19-bit storage dtype. Diagram by the author, based on [1](#ref-1), [3](#ref-3), [5](#ref-5), and [6](#ref-6).</figcaption>
</figure>

### 4.1 Normal numbers

For the IEEE binary interchange formats, when $0<E<2^w-1$, the represented value is

$$
x=(-1)^s 2^{E-B}\left(1+\frac{F}{2^t}\right),
$$

where the exponent bias is usually

$$
B=2^{w-1}-1.
$$

The leading $1$ is not stored. A normal binary value always has the form $1.f_1f_2\ldots f_t \times 2^e$, so recovering that predictable bit increases precision without increasing storage.

### 4.2 Zeros and subnormal numbers

The all-zero exponent field is reserved. If $E=0$ and $F=0$, the value is signed zero. If $E=0$ and $F\neq0$, the hidden leading digit becomes zero:

$$
x=(-1)^s 2^{1-B}\left(\frac{F}{2^t}\right).
$$

These are **subnormal** numbers. They fill the gap between zero and the smallest normal number with constant spacing, providing gradual underflow rather than an abrupt jump to zero. Implementations or performance modes can nevertheless flush subnormals to zero; that behavior must be documented separately from the abstract format.

### 4.3 Infinities and NaNs

For IEEE binary formats, an all-one exponent with a zero fraction represents signed infinity; an all-one exponent with a nonzero fraction represents NaN, or "not a number." NaNs allow invalid results such as $0/0$ to propagate without being confused with an ordinary finite number.

Not every low-precision ML format copies these special-value rules. The proposed FP8 E4M3 encoding extends its finite range by omitting infinities and reserving fewer patterns for NaNs, whereas E5M2 follows the IEEE convention more closely [5](#ref-5). "FP8" is therefore a family name, not a unique encoding.

## 5. Encoding a floating-point value by hand

Let us encode $13.25$ in IEEE binary16, commonly called FP16.

First convert the integer and fractional parts:

$$
13.25_{10}=1101.01_2.
$$

Normalize it:

$$
1101.01_2 = 1.10101_2\times2^3.
$$

Binary16 has five exponent bits, ten stored fraction bits, and bias $B=15$.

- The number is positive, so $s=0$.
- The unbiased exponent is $3$, so $E=3+15=18=10010_2$.
- Remove the implicit leading one and pad the fraction: $F=1010100000_2$.

The complete bit string is

```text
sign  exponent  fraction
 0     10010    1010100000
```

Grouping into hexadecimal digits gives `0x4AA0`. Decoding the fields returns

$$
(-1)^0 2^{18-15}\left(1+\frac{672}{1024}\right)
=8\times1.65625
=13.25.
$$

This value is exact because its fractional part is a power-of-two fraction. Decimal $0.1$ behaves differently.

### Why 0.1 repeats in binary

A rational number has a terminating base-two expansion only if its reduced denominator divides some power of two. Since

$$
0.1=\frac{1}{10}=\frac{1}{2\cdot5},
$$

and no power of two contains the factor 5, $0.1$ cannot terminate in binary. Its expansion repeats:

$$
0.1_{10}=0.00011001100110011\ldots_2.
$$

FP32 stores the nearest representable value under the usual round-to-nearest rule. Its bits are `0x3DCCCCCD`, whose exact decimal value is approximately $0.10000000149011612$. The datatype did not calculate the wrong real number; it selected the nearest member of a finite representable set.

## 6. Spacing, precision, and rounding

### 6.1 Spacing inside a binade

Consider normal positive numbers with the same unbiased exponent $e$. Consecutive fraction codes differ by one, so consecutive values differ by

$$
2^e\frac{1}{2^t}=2^{e-t}.
$$

The interval $[2^e,2^{e+1})$ is called a **binade**. Spacing is constant inside a binade and doubles when the exponent increases by one. Floating point therefore supplies approximately constant *relative* precision, not constant absolute precision.

<figure style="text-align: center">
  <img src="/images/posts/numerical-dtypes/float-spacing.svg" alt="Number lines showing dense floating-point values near one and values twice as far apart near two">
  <figcaption><b>Figure 3.</b> A toy format with three fraction bits. The spacing is $2^{-3}$ on $[1,2)$ and $2^{-2}$ on $[2,4)$. Diagram by the author.</figcaption>
</figure>

`numpy.finfo(dtype).eps`-style machine epsilon is usually the distance from $1$ to the next larger value:

$$
\epsilon = 2^{-t}.
$$

Numerical error analysis often uses the **unit roundoff**, half that distance for round-to-nearest:

$$
u=2^{-(t+1)}=2^{-p}.
$$

These two quantities are sometimes both called "machine epsilon," so a careful article or API should state which convention it uses.

### 6.2 A local relative-error bound

Suppose $x$ is a normal real number in $[2^e,2^{e+1})$, the rounded result is normal, and no overflow or underflow occurs. Adjacent floats in this binade are $2^{e-t}$ apart. Rounding to the nearest one introduces at most half that distance:

$$
|\operatorname{fl}(x)-x|\leq2^{e-t-1}.
$$

Because $|x|\geq2^e$,

$$
\frac{|\operatorname{fl}(x)-x|}{|x|}
\leq2^{-t-1}=u.
$$

Equivalently,

$$
\operatorname{fl}(x)=x(1+\delta),\qquad |\delta|\leq u.
$$

This compact model is powerful, but its conditions matter. It is not a universal relative bound near zero, under overflow, or for arbitrary non-IEEE approximations. Goldberg gives a detailed treatment of ulps, relative error, guard digits, and exact rounding [2](#ref-2).

### 6.3 Round to nearest, ties to even

IEEE 754 defines several rounding-direction attributes. The common default is round to nearest with ties resolved toward the result whose least significant digit is even [1](#ref-1). "Even" makes repeated halfway cases statistically less biased than always rounding upward.

Imagine a toy format with precision $p=4$. Near one, its values include

```text
1.000₂ = 1.000₁₀
1.001₂ = 1.125₁₀
```

The exact number $1.0625=1.0001_2$ lies halfway. The lower candidate has an even least-significant stored bit, so ties-to-even selects $1.000_2$.

## 7. How floating-point arithmetic is computed

An operation conceptually computes the exact real result and then rounds it to the destination format. Hardware obtains that result efficiently using extra guard, round, and sticky information rather than an infinitely wide register [2](#ref-2).

### 7.1 Addition and subtraction

To add two finite floating-point values:

1. compare exponents;
2. shift the smaller significand to align the binary points;
3. add or subtract the signed significands;
4. normalize the result;
5. round to the destination precision;
6. handle overflow, underflow, and special values.

For an exact example,

$$
1.5 = 1.10000_2\times2^0,
$$

$$
0.21875 = 1.11000_2\times2^{-3}
=0.00111_2\times2^0.
$$

After alignment,

$$
1.10000_2+0.00111_2=1.10111_2=1.71875.
$$

With enough fraction bits this sum is exact. With fewer bits the final significand must be rounded.

Alignment also explains **absorption**. If the exponents differ by more bits than the available precision, the smaller addend can be shifted entirely beyond the retained significand. Then $a+b$ may round to $a$ even when $b\neq0$.

Subtraction has an additional danger: if two approximate inputs are close, leading digits cancel and expose their earlier representation errors. This is catastrophic cancellation. Subtraction itself can still be correctly rounded; the problem is that the exact answer contains much less information than the operands appeared to contain.

### 7.2 Multiplication

For normal finite values written as

$$
x=(-1)^{s_x}m_x2^{e_x},\qquad
y=(-1)^{s_y}m_y2^{e_y},
$$

their exact product is

$$
xy=(-1)^{s_x\oplus s_y}(m_xm_y)2^{e_x+e_y}.
$$

The hardware XORs the sign bits, adds exponents after accounting for biases, multiplies significands, normalizes, and rounds. The unrounded significand product can need roughly twice the operand precision, which is one reason intermediate precision matters.

### 7.3 Fused multiply-add and accumulation

A fused multiply-add computes

$$
\operatorname{fma}(a,b,c)=\operatorname{round}(ab+c)
$$

with one final rounding, instead of rounding $ab$ first and then rounding the sum. The two expressions can produce different answers.

A dot product contains many multiply-adds:

$$
y=\sum_{i=1}^{n}a_ib_i.
$$

If every term were accumulated in a low-precision format, rounding would be introduced repeatedly. Using a wider accumulator usually reduces this error and extends the range of partial sums. This is why operand precision and accumulator precision must be reported separately in ML benchmarks.

Under the standard rounding model and assuming $nu<1$, error analyses of sequential inner products use the abbreviation

$$
\gamma_n=\frac{nu}{1-nu}.
$$

For a fused multiply-add sequence there is one rounding per term, leading to a $\gamma_n$-type bound. Separate rounded multiplications and additions require a correspondingly larger operation count, often expressed with $\gamma_{2n}$ in a simple analysis. The bounds grow approximately linearly with the operation count when that count times $u$ is small. They are worst-case statements, not predictions that every dot product loses that much accuracy; cancellation, data distribution, summation order, fused operations, and accumulator width all affect the observed result. Higham develops this notation and the inner-product bounds rigorously [10](#ref-10).

<figure style="text-align: center">
  <img src="/images/posts/numerical-dtypes/floating-addition.svg" alt="Floating-point addition pipeline showing exponent alignment, significand addition, normalization, and rounding">
  <figcaption><b>Figure 4.</b> The conceptual stages of floating-point addition. Individual processors may pipeline or combine these stages. Diagram by the author.</figcaption>
</figure>

## 8. The ML dtype landscape

The following table compares formats by their mathematical fields. The maximum finite and minimum normal values are properties of the stated encodings; actual libraries may restrict operations or use different handling for subnormals.

| Format | Sign / exponent / trailing bits | Precision $p$ | Approx. maximum finite | Approx. minimum positive normal | Unit roundoff $u$ |
|---|---:|---:|---:|---:|---:|
| FP64 | 1 / 11 / 52 | 53 | $1.80\times10^{308}$ | $2.23\times10^{-308}$ | $2^{-53}$ |
| FP32 | 1 / 8 / 23 | 24 | $3.40\times10^{38}$ | $1.18\times10^{-38}$ | $2^{-24}$ |
| TF32 compute | 1 / 8 / 10 | 11 | FP32-like range | FP32-like range | $2^{-11}$ |
| FP16 | 1 / 5 / 10 | 11 | $65504$ | $6.10\times10^{-5}$ | $2^{-11}$ |
| BF16 | 1 / 8 / 7 | 8 | $3.39\times10^{38}$ | $1.18\times10^{-38}$ | $2^{-8}$ |
| FP8 E4M3 | 1 / 4 / 3 | 4 | $448$ | $2^{-6}$ | $2^{-4}$ |
| FP8 E5M2 | 1 / 5 / 2 | 3 | $57344$ | $2^{-14}$ | $2^{-3}$ |

### 8.1 FP64 and FP32

FP64 spends 64 bits to provide a 53-bit significand and a much larger exponent range. It is essential in many scientific-computing problems, but doubling the bytes relative to FP32 increases storage and bandwidth requirements.

FP32 is the traditional baseline for neural-network training and general GPU numerical work. Its 24-bit precision gives roughly seven reliable decimal digits for a single rounded value, although an algorithm can lose more through conditioning or accumulated error.

### 8.2 FP16 versus BF16

FP16 and BF16 both occupy 16 bits, but allocate them differently:

- FP16 uses five exponent and ten trailing fraction bits. It has finer relative precision but a maximum finite value of 65504.
- BF16 uses eight exponent and seven trailing fraction bits. Its normal exponent range matches FP32, but its local spacing is eight times coarser than FP16 because $2^{-7}/2^{-10}=8$.

This is a clean example of the range-precision trade-off. The BF16 design and its deep-learning results are studied in Kalamkar et al. [3](#ref-3). Neither format is universally "more accurate": FP16 resolves nearby values more finely when they are in range; BF16 is much less likely to overflow or underflow at FP16's limits.

### 8.3 TF32 is a compute mode

NVIDIA's TensorFloat-32 uses an eight-bit exponent and ten explicitly represented fraction bits for selected tensor-core computations [6](#ref-6). Inputs and outputs are ordinarily held in FP32 storage. Calling TF32 a 19-bit tensor storage dtype is therefore misleading; it is better understood as reduced significand precision in a compute path with FP32-like exponent range.

Because backend settings can enable or disable TF32, two programs with FP32 tensors can execute matrix multiplications at different effective operand precision. Reproducible numerical work must record backend configuration as well as tensor dtypes.

### 8.4 FP8 is at least two formats

E4M3 spends more bits on the significand and is more precise locally. E5M2 spends more bits on the exponent and covers a much larger range. The joint NVIDIA, Arm, and Intel proposal reports E4M3 finite values up to 448 and E5M2 values up to 57344, with different special-value semantics [5](#ref-5).

At only three or four significant bits, unscaled conversion to FP8 can be severe. Practical systems use scaling recipes, higher-precision accumulation, and format selection by tensor role. Those policies are part of a quantization or mixed-precision system, not consequences of the eight-bit encoding alone.

### 8.5 FP4 and NF4 are not interchangeable

FP4 E2M1 is a tiny floating-point encoding with one sign, two exponent, and one trailing significand bit. NVIDIA's documented FP4 quantization scheme has finite values up to magnitude 6 and combines the codes with per-block scales [8](#ref-8).

NF4, introduced with QLoRA, instead uses a 16-value codebook chosen for approximately normally distributed weights [9](#ref-9). It is non-uniform: code points are not equally spaced, and its four bits are an index into a table rather than a sign/exponent/fraction decomposition. Both use four bits per code, but their mathematics and intended use differ.

### 8.6 Integers versus floats

For quick reference:

| Integer format | Exact code range | Adjacent code spacing | Real-value meaning |
|---|---:|---:|---|
| UINT8 | $[0,255]$ | 1 | Requires scale/zero-point for quantized real tensors |
| INT8 | $[-128,127]$ | 1 | Requires scale/zero-point for quantized real tensors |
| INT4 | $[-8,7]$ | 1 | Usually packed; requires grouping and scale metadata |

Integer grids have uniform spacing before scaling. Floating-point grids concentrate codes near zero and spread them out as magnitude grows. NF4 supplies a learned-by-design, distribution-aware codebook. Part 2 will compare these choices as quantizers rather than merely as bit layouts.

## 9. Three experiments you can reproduce

The companion script `experiments/numerical-dtypes/numerical_dtypes.py` uses only Python's standard library. Run it from the repository root:

```bash
python3 experiments/numerical-dtypes/numerical_dtypes.py
```

### Experiment 1: inspect 0.1

Python's `struct` module can round a value to IEEE binary32 storage and expose its bits:

```python
import struct

encoded = struct.pack(">f", 0.1)
bits = struct.unpack(">I", encoded)[0]
decoded = struct.unpack(">f", encoded)[0]

print(f"0x{bits:08X}")
print(f"{decoded:.17g}")
```

Expected output:

```text
0x3DCCCCCD
0.10000000149011612
```

This is a representation result derived from the FP32 format, not an empirical claim about a particular neural network.

### Experiment 2: addition is not associative

Using binary64 values,

```python
a = 1e16
b = -1e16
c = 1.0

print((a + b) + c)
print(a + (b + c))
```

The first expression produces `1.0`; the second commonly produces `0.0`. In the second parenthesization, adding `1.0` to `-1e16` is too small to change the stored value before $a$ is added. Real addition is associative; rounded floating-point addition is not.

### Experiment 3: summation strategy matters

The script compares an explicit left-to-right accumulation loop with `math.fsum`, which tracks partial sums more carefully. For `[1e16, 1.0, -1e16]`, the naive loop returns `0.0`, while `math.fsum` returns the exact result `1.0`. A more accurate algorithm can improve a result without changing the storage dtype. "Use more bits" is only one tool; stable formulations and accumulation strategies matter too.

These small experiments should not be generalized into performance claims. They isolate format behavior and numerical order, which is precisely what makes them reproducible across a broad range of systems.

## 10. A disciplined way to choose a dtype

Choosing a dtype is a constrained numerical-design problem. Ask, in order:

1. **Range:** Can inputs, intermediate values, and accumulators overflow or underflow?
2. **Resolution:** At the magnitudes of interest, is the spacing small enough to retain meaningful changes?
3. **Accumulation:** What dtype actually collects reductions and matrix products?
4. **Sensitivity:** Is the algorithm well-conditioned, or will small representation errors be amplified?
5. **Hardware path:** Does the target device execute this format natively, emulate it, or unpack it first?
6. **Evidence:** Has the complete workload been compared with a higher-precision reference?

The last question is indispensable. A dtype can satisfy a local error bound while the full algorithm remains inaccurate because the problem is ill-conditioned. Conversely, a neural network can tolerate surprisingly coarse local representations because its learned computation is robust to them. Bit counts alone prove neither outcome.

## 11. What this establishes for quantization

We now have the pieces needed to analyze quantization without hand-waving:

- an integer code has a finite exact range;
- a scale and zero-point map integer codes to a uniform real grid;
- floating-point spacing changes with exponent;
- rounding error can be bounded locally when its assumptions hold;
- storage, multiplication, and accumulation can use different precisions;
- fewer bits do not imply proportional speedup without a matching hardware path.

Part 2 will use these facts to derive affine quantization, clipping and rounding error, per-tensor versus groupwise scales, and the logic behind methods such as LLM.int8(), SmoothQuant, GPTQ, AWQ, and NF4/QLoRA.

## References

1. <a id="ref-1"></a>IEEE Standards Association. [IEEE 754-2019: IEEE Standard for Floating-Point Arithmetic](https://standards.ieee.org/ieee/754/6210/), 2019.
2. <a id="ref-2"></a>David Goldberg. [What Every Computer Scientist Should Know About Floating-Point Arithmetic](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html). *ACM Computing Surveys*, 23(1), 1991.
3. <a id="ref-3"></a>Dhiraj Kalamkar et al. [A Study of BFLOAT16 for Deep Learning Training](https://arxiv.org/abs/1905.12322), 2019.
4. <a id="ref-4"></a>Shibo Wang and Pankaj Kanwar. [BFloat16: The Secret to High Performance on Cloud TPUs](https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus), Google Cloud, 2019.
5. <a id="ref-5"></a>Paulius Micikevicius et al. [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433), 2022.
6. <a id="ref-6"></a>NVIDIA. [TensorRT: Accuracy Considerations](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/accuracy-considerations.html), accessed 2026-08-27.
7. <a id="ref-7"></a>PyTorch. [Numerical Accuracy](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html), accessed 2026-08-27.
8. <a id="ref-8"></a>NVIDIA. [TensorRT Quantization Schemes](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/quantized-types-schemes.html), accessed 2026-08-27.
9. <a id="ref-9"></a>Tim Dettmers et al. [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314), 2023.
10. <a id="ref-10"></a>Nicholas J. Higham. [Accuracy and Stability of Numerical Algorithms, Chapter 3: Basics](https://epubs.siam.org/doi/10.1137/1.9780898718027.ch3), second edition, SIAM, 2002.
