# -*- coding: utf-8 -*-
import openai
import json
from tqdm import tqdm
import time

openai.api_key = "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"  # 粘贴API
json_name = "prompts_imagenet_5.json"  # 记得每次给序号+1，保障顺序不要乱

category_list_cn = ['橘子', '柠檬', '无花果', '菠萝', '香蕉', '菠萝蜜', '番荔枝', '石榴', '干草', '培根蛋酱意大利面', '巧克力酱', '面团', '肉卷', '披萨', '馅饼', '卷饼', '红葡萄酒', '意式浓缩咖啡', '茶杯', '蛋酒', '高山', '泡泡', '悬崖', '珊瑚礁', '间歇泉', '湖边', '岬角', '沙洲', '沙滩', '峡谷', '火山', '棒球运动员', '新郎', '潜水员', '油菜', '雏菊', '黄色杓兰', '玉米', '橡子', '玫瑰果', '七叶树果实', '珊瑚菌', '木耳', '鹿花菌', '臭角菇', '地星', '野生舞茸', '牛肝菌', '玉米棒子', '卫生纸']
category_list_en = ['orange', 'lemon', 'fig', 'pineapple', 'banana', 'jackfruit', 'cherimoya (custard apple)', 'pomegranate', 'hay', 'carbonara', 'chocolate syrup', 'dough', 'meatloaf', 'pizza', 'pot pie', 'burrito', 'red wine', 'espresso', 'tea cup', 'eggnog', 'mountain', 'bubble', 'cliff', 'coral reef', 'geyser', 'lakeshore', 'promontory', 'sandbar', 'beach', 'valley', 'volcano', 'baseball player', 'bridegroom', 'scuba diver', 'rapeseed', 'daisy', "yellow lady's slipper", 'corn', 'acorn', 'rose hip', 'horse chestnut seed', 'coral fungus', 'agaric', 'gyromitra', 'stinkhorn mushroom', 'earth star fungus', 'hen of the woods mushroom', 'bolete', 'corn cob', 'toilet paper']

# 下面地代码都别动

all_responses = {}
num_response = 5

for i in tqdm(range(len(category_list_cn))):

    prompts = ["用中文描述" + category_list_cn[i] + "（" + category_list_en[i] + "）" + "的样子。",
               "用中文简洁地描述" + category_list_cn[i] + "（" + category_list_en[i] + "）" + "的样子。",
               "用中文描述一张" + category_list_cn[i] + "（" + category_list_en[i] + "）" + "的图片。"]

    all_result = []

    prompt_idx = 0
    while prompt_idx < len(prompts):

        try:
            curr_prompt = prompts[prompt_idx]
            completion = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                                      messages=[{"role": "user", "content": "%s" % curr_prompt}],
                                                      n=num_response,
                                                      max_tokens=55,
                                                      temperature=0.99)
            for j in range(num_response):
                result = completion.choices[j].message.content
                all_result.append(result.replace("\n\n", "") + ".")
            prompt_idx += 1
        except openai.error.RateLimitError:
            time.sleep(10)  # sleep for 10 secs, and go back to the original prompt

    all_responses[category_list_cn[i]] = all_result

with open(json_name, 'w', encoding="utf-8") as f:
    json.dump(all_responses, f, indent=4, ensure_ascii=False)
