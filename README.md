# covid-19
Covid-19 project based on Adrian Rosebrock Code

In this tutorial, you will learn how to **automatically detect COVID-19 in a
 hand-created X-ray image dataset** using Keras, TensorFlow, and Deep Learning.

Like most people in the world right now, I’m genuinely concerned about COVID-19. I find myself constantly analyzing my personal health and wondering if/when I will contract it.

The more I worry about it, the more it turns into a painful mind game of legitimate symptoms combined with hypochondria:

I woke up this morning feeling a bit achy and run down.
As I pulled myself out of bed, I noticed my nose was running (although it’s now reported that a runny nose is not a symptom of COVID-19).
By the time I made it to the bathroom to grab a tissue, I was coughing as well.
At first, I didn’t think much of it — I have pollen allergies and due to the warm weather on the eastern coast of the United States, spring has come early this year. My allergies were likely just acting up.

But my symptoms didn’t improve throughout the day.

I’m actually sitting here, writing the this tutorial, with a thermometer in my mouth; and glancing down I see that it reads 99.4° Fahrenheit.

My body runs a bit cooler than most, typically in the 97.4°F range. Anything above 99°F is a low-grade fever for me.

Cough and low-grade fever? That could be COVID-19…or it could simply be my allergies.

It’s impossible to know without a test, and that “not knowing” is what makes this situation so scary from a visceral human level.

As humans, there is nothing more terrifying than the unknown.

Despite my anxieties, I try to rationalize them away. I’m in my early 30s, very much in shape, and my immune system is strong. I’ll quarantine myself (just in case), rest up, and pull through just fine — COVID-19 doesn’t scare me from my own personal health perspective (at least that’s what I keep telling myself).

That said, I am worried about my older relatives, including anyone that has pre-existing conditions, or those in a nursing home or hospital. They are vulnerable and it would be truly devastating to see them go due to COVID-19.

Instead of sitting idly by and letting whatever is ailing me keep me down (be it allergies, COVID-19, or my own personal anxieties), I decided to do what I do best — focus on the overall CV/DL community by writing code, running experiments, and educating others on how to use computer vision and deep learning in practical, real-world applications.

That said, I’ll be honest, this is not the most scientific article I’ve ever written. Far from it, in fact. The methods and datasets used would not be worthy of publication. But they serve as a starting point for those who need to feel like they’re doing something to help.

I care about you and I care about this community. I want to do what I can to help — this blog post is my way of mentally handling a tough time, while simultaneously helping others in a similar situation.

I hope you see it as such.

Inside of today’s tutorial, you will learn how to:

1. Sample an open source dataset of X-ray images for patients who have tested
 positive for COVID-19
2. Sample “normal” (i.e., not infected) X-ray images from healthy patients
3. Train a CNN to automatically detect COVID-19 in X-ray images via the dataset
 we created
4. Evaluate the results from an educational perspective

###How could COVID-19 be detected in X-ray images?

https://www.pyimagesearch.com/wp-content/uploads/2020/03/covid19_keras_xray_example.jpg

COVID-19 tests are currently hard to come by — [there are simply not enough of
 them and they cannot be manufactured fast enough](https://www.usatoday.com/story/news/2020/03/11/coronavirus-covid-19-response-hurt-by-shortage-testing-components/5013586002/), which is causing panic.
 
 When there’s panic, [there are nefarious people looking to take advantage of
  others, namely by selling fake COVID-19 test kits](https://abc7news.com/5995593/) after [finding victims on social media platforms and chat
   applications](https://www.edgeprop.my/content/1658343/covid-19-home-testing-kits-are-fake-medical-authority).
  
 Given that there are limited COVID-19 testing kits, we need to rely on other diagnosis measures.
 
 For the purposes of this tutorial, I thought to explore X-ray images as doctors frequently use X-rays and CT scans to diagnose pneumonia, lung inflammation, abscesses, and/or enlarged lymph nodes.
 
 Since COVID-19 attacks the epithelial cells that line our respiratory tract, we can use X-rays to analyze the health of a patient’s lungs.
 
 And given that nearly all hospitals have X-ray imaging machines, it could be possible to use X-rays to test for COVID-19 without the dedicated test kits.
 
 A drawback is that X-ray analysis requires a radiology expert and takes
  significant time — which is precious when people are sick around the world
  . **Therefore developing an automated analysis system is required to save
   medical professionals valuable time**.
 
 **Note**: There are newer publications that suggest CT scans are better for
  diagnosing COVID-19, but all we have to work with for this tutorial is an X-ray image dataset. Secondly, I am not a medical expert and I presume there are other, more reliable, methods that doctors and medical professionals will use to detect COVID-19 outside of the dedicated test kits. 
  
 ### Our COVID-19 patient X-ray image dataset 
 
 **Step 1: to curate X-ray images of COVID-19 patients.**
 
 The COVID-19 X-ray image dataset we’ll be using for this tutorial was
  curated by [Dr. Joseph Cohen](https://josephpcohen.com/w/), a postdoctoral
   fellow at the University of Montreal.
   
 One week ago, Dr. Cohen started collecting X-ray images of COVID-19 cases
  and publishing them in the [following GitHub repo](https://github.com
  /ieee8023/covid-chestxray-dataset). [Paper: COVID-19 Image Data Collection
  ](https://arxiv.org/pdf/2003.11597)
  
 1. Inside the repo you’ll find example of COVID-19 cases, as well as MERS
 , SARS, and ARDS.
 
 2. In order to create the COVID-19 X-ray image dataset for this tutorial, I:
 
    1. Parsed the metadata.csv file found in Dr. Cohen’s repository.
 Selected all rows that are:
 Positive for COVID-19 (i.e., ignoring MERS, SARS, and ARDS cases).
    2. Posterioranterior (PA) view of the lungs. I used the PA view as, to my
  knowledge, that was the view used for my “healthy” cases, as discussed below; however, I’m sure that a medical professional will be able clarify and correct me if I am incorrect (which I very well may be, this is just an example).
  
 In total, that left me with **25 X-ray images of positive COVID-19 cases**
  (Figure 2, left). 
  
 **Step 2: to curate X-ray images of healthy patients.**
 
 To do so, I used [Kaggle’s Chest X-Ray Images (Pneumonia) dataset](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) and
  sampled 25 X-ray images from healthy patients (Figure 2, right). There are a number of problems with Kaggle’s Chest X-Ray dataset, namely noisy/incorrect labels, but it served as a good enough starting point for this proof of concept COVID-19 detector.
 
 After gathering my dataset, I was left with 50 total images, equally split with 25 images of COVID-19 positive X-rays and 25 images of healthy patient X-rays.
 
 I’ve included my sample dataset in the “Downloads” section of this tutorial, so you do not have to recreate it.
 
 Additionally, I have included my Python scripts used to generate the dataset in the downloads as well, but these scripts will not be reviewed in this tutorial as they are outside the scope of the post.
 
 