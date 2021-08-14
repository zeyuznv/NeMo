import sys
sys.path = ["/home/lab/NeMo"] + sys.path
from nemo.collections.nlp.models import PunctuationCapitalizationModel

# to get the list of pre-trained models
PunctuationCapitalizationModel.list_available_models()

# Download and load the pre-trained BERT-based model
model = PunctuationCapitalizationModel.from_pretrained("punctuation_en_bert")

text = "i had about five minutes before i was set to deliver a talk to a bunch of business owners about visibility and being on camera after all i was the so called expert there the former two year television news anchor and life and business coach i happen to take a look down at my cell phone just to catch the time and i noticed that i had a miss call from my ex husband i can still hear his voice darius what is going on i just got a call from some strange man who told me to go to this website and now i'm looking at all of these photos of you naked your private parts are all over this website who's seen this i couldn't think i couldn't breathe i was so humiliated and so embarrassed and so ashamed i felt like my world was coming to an end and yet this began for me months of pain and depression and anger and confusion in silence my manipulative jealous stalker ex boyfriend did exactly what he said he would do he put up a website with my name on it and he posted this and this and several explicit photos that he had taken of me while i was asleep living with him in jamaica for months prior to that he had been sending me threatening text messages like this he was trying to make me out to be some sleeazy low life flut he'd even threaten to kill me he told me that he would shoot me in my head and stab me at my heart simply because i wanted to end the controlling relationship i couldn't believe this was happening to me i didn't even know what to call it you might know it as cyber harassment or cyber bullying the media calls it revenge porn i now call it digital domestic violence it typically stems from a relationship gone bad where a controlling gilted ex lover can't handle rejection so when they can't physically put their hands on you they use different weapons cell phones and laptops the ammunition photos videos explicit information content all posted online without your consent i mean let's face it we all live our lives online and the internet is a really small world we show off our baby photos we start and grow our businesses we make new relationships we let the world in one facebook like at a time and you know what i found an even smaller world one in 25 women say they have been impacted by revenge porn for women under the age of 30 that number looks like one in 10 and that leaves a few of you in this audience as potential victims you want to know what's even more alarming lack of legislation and laws to adequately protect victims and punish perpetrators there's only one federal bill pending it's called the enough act by senator kamalla harris it would criminalize revenge porn but that could take years to pass so what are we left with in the meantime flimsy civil misdemeanors currently only 40 states and dc have some laws in place for revenge porn and those penalties vary we're talking 500 dollars fines 500 dollars are you kidding me women are losing their jobs they're suffering from damaged relationships and damaged reputations they're falling into illness and depression and the suicide rates are climbing you're looking at a woman who spent 11 months in court 13 trips to the courthouse and thousands of dollars in legal fees just to get two things a protection from cyber stalking and cyber abuse otherwise known as a pfa and language from a judge that would force a 3rd party internet company to remove the content it's expensive complicated and confusing and worse legal loopholes and jurisdictional issues dragged this out for months while my private parts were on display for months how would you feel if your naked body was exposed for the world to see and you waited helplessly for the content to be removed eventually i stumbled upon a private company to issue a dmca notice to shut the website down dmca digital millennial copyright act it's a law that regulates digital material and content broadly the aim of the dmca is to protect both copyright owners and consumers so get this people who take and share nude photos own the rights to those selfies so they should be able to issue a dmca to have the content removed but not so fast because the other fight we're dealing with is non compliant and non responsive 3rd party internet companies and oh by the way even in consenting relationships just because you get a nude photo or a naked pick does not give you the right to share it even with the intent to do harm back to my case which happens to be further complicated because he was stalking and harassing me from another country making it nearly impossible to get help here but wait a minute isn't the internet international shouldn't we have some sort of policy in place that broadly protects us regardless to borders or restrictions i just couldn't give up i had to keep fighting so i willingly on three occasions allowed for the invasion of both my cell phone and my laptop by the department of homeland security and the jamaican embassy for thorough forensic investigation because i had maintained all of the evidence i painstakingly shared my private parts with the all mail investigative team and it was an embarrassing humiliating additional hoop to jump through but then something happened jamaican authorities actually arrested him he's now facing charges under their malicious communications act and if found guilty could face thousands of dollars in fines and up to 10 years in prison and i've also learned that my case is making history it is the 1st international case under this new crime wow finally some justice but this got me to thinking nobody deserves this nobody deserves this level of humiliation and having to jump through all of these hoops our cyber civil rights are at stake here in the united states we need to have clear tough enforcement we need to demand the accountability and responsiveness from online companies we need to promote social responsibilities for posting sharing and texting and we need to restore dignity to victims and what about victims who neither have the time money or resources to wage war who are left disempowered mislabellled and broken two things release the shame and in the silence shame is at the core of all of this and for every silent prisoner of shame it's the fear of judgment that's holding you hostage and the price to pay is the stripping away of your self worth the day i ended my silence i freed myself from shame and i freed myself from the fear of judgment from the one person who i thought would judge me the most my son who actually told me mom you are the strongest person that i know you can get through this and besides mom he chose the wrong woman to mess with it was on that day that i decided to use my platform and my story and my voice and to get started i asked myself this one simple question who do i need to become now that question in the face of everything that i was challenged with transform my life and had me thinking about all kinds of possibilities i now own my story i speak my truth and i'm narrating a new chapter in my life it's called 50 shades of silence it's a global social justice project and we're working to film an upcoming documentary to give voice and dignity to victims if you are a victim or you know someone who is know this in order to be empowered you have to take care of yourself and you have to love yourself you have to turn your anger into action your pain into power and your setback into a setup for what's next for your life this is a process and it's a journey of self discovery that might include forgiveness but it definitely requires bravery confidence and conviction i call it finding your everyday courage thank you"

# try the model on a few examples

texts = [
    "",
    "a",
    "of",
    "foo",
    'how are you',
    'great how about you',
    text
]

for m in [0, 1, 2, 3]:
    print(f"margin: {m}")
    try:
        res = model.add_punctuation_capitalization(texts, step=510, margin=m)
    except ValueError as e:
        print(e)
        continue
    for i, r in enumerate(res):
        print(f"res[{i}]:", repr(res[i]))
for m in [0, 1, 2, 3]:
    for s in [1, 2, 3, 4]:
        for l in [3, 4, 5, 6]:
            try:
                res = model.add_punctuation_capitalization(texts, step=s, margin=m, max_seq_length=l)
            except ValueError as e:
                print(e)
                continue
            for i, r in enumerate(res):
                print(f"res[{i}]:", repr(res[i]))

