from utils import init
import json
import os
import logging
import sys
import pandas as pd
def analyze_random_results(synonym_data,antonym_data):
    synonym_num=0
    synonym_miss=0
    antonym_num=0
    antonym_miss=0
    synonym_diff=[]
    antonym_diff=[]
    synonym_diff_id=[]
    antonym_diff_id=[]
    synonym_att_list=[]
    synonym_percentage_list=[]
    antonym_att_list=[]
    antonym_percentage_list=[]
    for i in range(len(synonym_data)):
        synonym_att_list.append(synonym_data[i]['change_attention'])
        antonym_att_list.append(antonym_data[i]['change_attention'])
        synonym_percentage_list.append(synonym_data[i]['change_percentage'])
        antonym_percentage_list.append(antonym_data[i]['change_percentage'])
        #if (synonym_data[i]['prediction']!=None) and (synonym_data[i]['previous_prediction']!=None):
        if synonym_data[i]['previous_prediction']!=None:
            if(synonym_data[i]['prediction']!=synonym_data[i]['previous_prediction']):
                synonym_num+=1
                synonym_diff.append(synonym_data[i])
                synonym_diff_id.append(i)
        else:
            synonym_miss+=1 
        #if (antonym_data[i]['prediction']!=None) and (antonym_data[i]['previous_prediction']!=None):
        if antonym_data[i]['previous_prediction']!=None:
            if(antonym_data[i]['prediction']!=antonym_data[i]['previous_prediction']):
                antonym_num+=1
                antonym_diff.append(antonym_data[i])
                antonym_diff_id.append(i)
        else:
            antonym_miss+=1 
    return synonym_num,antonym_num,synonym_diff,antonym_diff,synonym_miss,antonym_miss
def analyze_top_results(synonym_data,antonym_data,mode):
    synonym_num=0
    antonym_num=0
    synonym_miss=0
    antonym_miss=0
    synonym_diff=[]
    antonym_diff=[]
    synonym_diff_id=[]
    antonym_diff_id=[]
    synonym_list_before=[]
    synonym_percentage_list_before=[]
    antonym_list_before=[]
    antonym_percentage_list_before=[]
    synonym_list_after=[]
    synonym_percentage_list_after=[]
    antonym_list_after=[]
    antonym_percentage_list_after=[]
    for i in range(len(synonym_data)):
        synonym_list_before.append(synonym_data[i]['att/grad_value'])
        antonym_list_before.append(antonym_data[i]['att/grad_value'])
        synonym_percentage_list_before.append(synonym_data[i]['att/grad_in_CoT'])
        antonym_percentage_list_before.append(antonym_data[i]['att/grad_in_CoT'])
        #if mode=='att':
            #synonym_list_after.append(synonym_data[i]['att_percentage_in_CoT_after_replacing'])
            #antonym_list_after.append(antonym_data[i]['att_percentage_in_CoT_after_replacing'])
            #synonym_percentage_list_after.append(synonym_data[i]['att_after_replacing'])
            #antonym_percentage_list_after.append(antonym_data[i]['att_after_replacing'])
        # else:
        #     synonym_list_after.append(synonym_data[i]['grad_percentage_in_CoT_after_replacing'])
        #     antonym_list_after.append(antonym_data[i]['grad_percentage_in_CoT_after_replacing'])
        #     synonym_percentage_list_after.append(synonym_data[i]['grad_after_replacing'])
        #     antonym_percentage_list_after.append(antonym_data[i]['grad_after_replacing'])
        #if (synonym_data[i]['prediction']!=None) and (synonym_data[i]['previous_prediction']!=None):
        if synonym_data[i]['previous_prediction']!=None:
            if(synonym_data[i]['prediction']!=synonym_data[i]['previous_prediction']):
                synonym_num+=1
                synonym_diff.append(synonym_data[i])
                synonym_diff_id.append(i)
        else:
            synonym_miss+=1 
        #if (antonym_data[i]['prediction']!=None) and (antonym_data[i]['previous_prediction']!=None):
        if antonym_data[i]['previous_prediction']!=None:
            if(antonym_data[i]['prediction']!=antonym_data[i]['previous_prediction']):
                antonym_num+=1
                antonym_diff.append(antonym_data[i])
                antonym_diff_id.append(i)
        else:
            antonym_miss+=1 
    return synonym_num,antonym_num,synonym_diff,antonym_diff,synonym_miss,antonym_miss

def simple_analyze(data):
    num=0
    miss=0
    diff=[]
    diff_id=[]
    for i in range(len(data)):
        #if (data[i]['prediction']!=None) and (data[i]['previous_prediction']!=None):
        if data[i]['previous_prediction']!=None:
            if(data[i]['prediction']!=data[i]['previous_prediction']):
                num+=1
                diff.append(data[i])
                diff_id.append(i)
        else:
            miss+=1
    return num,diff,miss

if __name__=='__main__':
    #result_record=[]
    root='.'
    method={
    0:'tsne',
    1:'umap'
    }[1]
    data_name,data_path,sub_set,model_name,model_path,output_path=init()
    log_file = f"./log/{model_name}-analyze-{data_name}_{sub_set}.log"
    log_dir = os.path.dirname(log_file)
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    rate=1
    #读取random数据
    with open(f"{root}/final_results/replace_results/{model_name}-{data_name}-{sub_set}-synonym-rate{rate}_random_right.json") as f:
        synonym_data=json.load(f)
    with open(f"{root}/final_results/replace_results/{model_name}-{data_name}-{sub_set}-antonym-rate{rate}_random_right.json") as f:
        antonym_data=json.load(f)
    synonym_num,antonym_num,synonym_diff,antonym_diff,synonym_miss,antonym_miss=analyze_random_results(synonym_data,antonym_data)
    logging.info("="*10+"Random Results"+'='*10)
    logging.info('synonym_rate:{:.1f}%,synonym_change:{}/{},miss{}'.format((synonym_num/len(synonym_data))*100,synonym_num,len(synonym_data),synonym_miss))
    logging.info('antonym_rate:{:.1f}%,antonym_change:{}/{},miss{}'.format((antonym_num/len(antonym_data))*100,antonym_num,len(antonym_data),antonym_miss))
    #result_record.append('{}/{}'.format(synonym_num,len(synonym_data)))
    #存储diff数据
    os.makedirs(f'{root}/final_results/diff/random/', exist_ok=True)
    with open(f'{root}/final_results/diff/random/{model_name}-{data_name}-{sub_set}-synonym-rate{rate}_right_diff.json', 'w', encoding='utf-8') as f:
        json.dump(synonym_diff, f, ensure_ascii=False, indent=4)
    with open(f'{root}/final_results/diff/random/{model_name}-{data_name}-{sub_set}-antonym-rate{rate}_right_diff.json', 'w', encoding='utf-8') as f:
        json.dump(antonym_diff, f, ensure_ascii=False, indent=4)
    #读取top数据
    mode_list=['att']
    top_k_percentage_list=[0.1,0.2,0.3]
    for top_k_percentage in top_k_percentage_list:
        for mode in mode_list:
        #for top_k_percentage in top_k_percentage_list:
            if mode=='att':
                with open(f"{root}/final_results/replace_results/{model_name}-{data_name}-{sub_set}-synonym-{mode}-right-top_percentage{top_k_percentage}.json") as f:
                    synonym_data=json.load(f)
                with open(f"{root}/final_results/replace_results/{model_name}-{data_name}-{sub_set}-antonym-{mode}-right-top_percentage{top_k_percentage}.json") as f:
                    antonym_data=json.load(f)
            else:
                with open(f"{root}/final_results/replace_results/{model_name}-{data_name}-{sub_set}-synonym-{mode}-correct-right-top_percentage{top_k_percentage}.json") as f:
                    synonym_data=json.load(f)
                with open(f"{root}/final_results/replace_results/{model_name}-{data_name}-{sub_set}-antonym-{mode}-correct-right-top_percentage{top_k_percentage}.json") as f:
                    antonym_data=json.load(f)
            synonym_num,antonym_num,synonym_diff,antonym_diff,synonym_miss,antonym_miss=analyze_top_results(synonym_data,antonym_data,mode)
            logging.info("="*10+f"Top Results mode={mode},top_k_percentage={top_k_percentage}"+'='*10)
            logging.info('synonym_rate:{:.1f}%,synonym_change:{}/{},miss{}'.format((synonym_num/len(synonym_data))*100,synonym_num,len(synonym_data),synonym_miss))
            logging.info('antonym_rate:{:.1f}%,antonym_change:{}/{},miss{}'.format((antonym_num/len(antonym_data))*100,antonym_num,len(antonym_data),antonym_miss))
            #存储top diff数据
            if mode=='att':
                os.makedirs(f'{root}/final_results/diff/att/', exist_ok=True)
                with open(f'{root}/final_results/diff/att/{model_name}-{data_name}-{sub_set}-synonym-{mode}-right-top_percentage{top_k_percentage}.json', 'w', encoding='utf-8') as f:
                        json.dump(synonym_diff, f, ensure_ascii=False, indent=4)
                with open(f'{root}/final_results/diff/att/{model_name}-{data_name}-{sub_set}-antonym-{mode}-right-top_percentage{top_k_percentage}.json', 'w', encoding='utf-8') as f:
                        json.dump(antonym_diff, f, ensure_ascii=False, indent=4)
            if mode=='grad':
                os.makedirs(f'{root}/final_results/diff/grad/', exist_ok=True)
                with open(f'{root}/final_results/diff/grad/{model_name}-{data_name}-{sub_set}-synonym-{mode}-correct-right-top_percentage{top_k_percentage}.json', 'w', encoding='utf-8') as f:
                        json.dump(synonym_diff, f, ensure_ascii=False, indent=4)
                with open(f'{root}/final_results/diff/grad/{model_name}-{data_name}-{sub_set}-antonym-{mode}-correct-right-top_percentage{top_k_percentage}.json', 'w', encoding='utf-8') as f:
                        json.dump(antonym_diff, f, ensure_ascii=False, indent=4)
    #读取recover from mistakes数据
    rate=1
    mode_list_2=['att','random']
    input_path=f"{root}/final_results/recover_results/"
    #for top_k_percentage in top_k_percentage_list:
    for mode in mode_list_2:
        for top_k_percentage in top_k_percentage_list:
            if mode=='att':
                with open(input_path+f"{model_name}-{data_name}-{sub_set}-antonym-{mode}-right-top_percentage{top_k_percentage}.json") as f:
                    data=json.load(f)
            elif mode=='grad':
                with open(input_path+f"{model_name}-{data_name}-{sub_set}-antonym-{mode}-correct-right-top_percentage{top_k_percentage}.json") as f:
                    data=json.load(f)
            else:
                with open(input_path+f"{model_name}-{data_name}-{sub_set}-antonym-rate{rate}_{mode}_right.json") as f:
                    data=json.load(f)
            num,diff,miss=simple_analyze(data)
            #保存结果
            output_path=f"{root}/final_results/diff/recover/"
            os.makedirs(output_path,exist_ok=True)
            logging.info("="*10+f"Recover From Mistakes mode={mode},top_k_percentage={top_k_percentage}"+'='*10)
            logging.info('rate:{:.1f}%,change:{}/{},miss{}'.format((num/len(data))*100,num,len(data),miss))
            if mode=='att':
                with open(output_path+f"{model_name}-{data_name}-{sub_set}-antonym-{mode}-right-top_percentage{top_k_percentage}.json",'w', encoding='utf-8') as f:
                    json.dump(diff, f, ensure_ascii=False, indent=4)
            elif mode=='grad':
                with open(output_path+f"{model_name}-{data_name}-{sub_set}-antonym-{mode}-correct-right-top_percentage{top_k_percentage}.json",'w', encoding='utf-8') as f:
                    json.dump(diff, f, ensure_ascii=False, indent=4)
            else:
                with open(output_path+f"{model_name}-{data_name}-{sub_set}-rate{rate}_{mode}_right.json",'w', encoding='utf-8') as f:
                    json.dump(diff, f, ensure_ascii=False, indent=4)
    #读取maintain_only_words数据
    mode_list=['att']
    mask_file={0:'mask',1:'unmask'}[1]
    filter_flag_list=[True]
    for top_k_percentage in top_k_percentage_list:
        for mode in mode_list:
            for filter_flag in filter_flag_list:
            #for top_k_percentage in top_k_percentage_list:
                if filter_flag:
                    input_path=f"{root}/final_results/maintain_only_topk/filter/"
                    input_path_rest=f'{root}/final_results/maintain_rest/filter/'
                    input_path_sampled=f'{root}/final_results/maintain_sampled/filter/'
                    input_path_knn=f'{root}/final_results/maintain_KNN_{method}/filter/'
                    input_path_low=f'{root}/final_results/maintain_low/filter/'
                    input_path_sampled_all=f'{root}/final_results/maintain_sampled_all/filter/'
                else:
                    input_path=f"{root}/final_results/maintain_only_topk/"
                    input_path_rest=f'{root}/final_results/maintain_rest/'
                    input_path_sampled=f'{root}/final_results/maintain_sampled/'
                    input_path_knn=f'{root}/final_results/maintain_KNN_{method}/'
                    input_path_low=f'{root}/final_results/maintain_low/'
                    input_path_sampled_all=f'{root}/final_results/maintain_sampled_all/'
                if mode=='att':
                    with open(input_path+f"{mask_file}/{model_name}-{data_name}-{sub_set}-{mode}-right-top_percentage{top_k_percentage}.json") as f:
                        data=json.load(f)
                    with open(input_path_rest+f"{mask_file}/{model_name}-{data_name}-{sub_set}-{mode}-right-top_percentage{top_k_percentage}.json") as f:
                        data_rest=json.load(f)
                    with open(input_path_sampled+f"{mask_file}/{model_name}-{data_name}-{sub_set}-{mode}-right-top_percentage{top_k_percentage}.json") as f:
                        data_sampled=json.load(f)
                    with open(input_path_knn+f"{mask_file}/{model_name}-{data_name}-{sub_set}-knn-last-top_percentage{top_k_percentage}.json") as f:
                        data_knn_last=json.load(f)
                    with open(input_path_knn+f"{mask_file}/{model_name}-{data_name}-{sub_set}-knn-pass-top_percentage{top_k_percentage}.json") as f:
                        data_knn_pass=json.load(f)
                    with open(input_path_low+f"{mask_file}/{model_name}-{data_name}-{sub_set}-{mode}-right-top_percentage{top_k_percentage}.json")as f:
                        data_low=json.load(f)
                    with open(input_path_sampled_all+f"{mask_file}/{model_name}-{data_name}-{sub_set}-{mode}-right-top_percentage{top_k_percentage}.json")as f:
                        data_sampled_all=json.load(f)
                elif mode=='grad':
                    with open(input_path+f"{mask_file}/{model_name}-{data_name}-{sub_set}-{mode}-correct-right-top_percentage{top_k_percentage}.json") as f:
                        data=json.load(f)
                    with open(input_path_rest+f"{mask_file}/{model_name}-{data_name}-{sub_set}-{mode}-correct-right-top_percentage{top_k_percentage}.json") as f:
                        data_rest=json.load(f)
                    with open(input_path_sampled+f"{mask_file}/{model_name}-{data_name}-{sub_set}-{mode}-correct-right-top_percentage{top_k_percentage}.json") as f:
                        data_sampled=json.load(f)
                else:
                    with open(input_path+f"{mask_file}/{model_name}-{data_name}-{sub_set}-{mode}-grad_correct-right-top_percentage{top_k_percentage}.json") as f:
                        data=json.load(f)
                    with open(input_path_rest+f"{mask_file}/{model_name}-{data_name}-{sub_set}-{mode}-grad_correct-right-top_percentage{top_k_percentage}.json") as f:
                        data_rest=json.load(f)
                    with open(input_path_sampled+f"{mask_file}/{model_name}-{data_name}-{sub_set}-{mode}-grad_correct-right-top_percentage{top_k_percentage}.json") as f:
                        data_sampled=json.load(f)
                num,diff,miss=simple_analyze(data)
                num_rest,diff_rest,miss_rest=simple_analyze(data_rest)
                num_sampled,diff_sampled,miss_sampled=simple_analyze(data_sampled)
                num_knn_last,diff_knn_last,miss_knn_last=simple_analyze(data_knn_last)
                num_knn_pass,diff_knn_pass,miss_knn_pass=simple_analyze(data_knn_pass)
                num_low,diff_low,miss_low=simple_analyze(data_low)
                num_sampled_all,diff_sampled_all,miss_sampled_all=simple_analyze(data_sampled_all)
                logging.info("="*10+f"Maintain Only mode={mode},filter_flag={filter_flag},top_k_percentage={top_k_percentage}"+'='*10)
                logging.info('maitain_only High-ATT rate:{:.1f}%,correct:{}/{},miss{}'.format((1-num/len(data))*100,len(data)-num,len(data),miss))
                logging.info('maitain_only Low-ATT rate:{:.1f}%,correct:{}/{},miss{}'.format((1-num_low/len(data_low))*100,len(data_low)-num_low,len(data_low),miss_low))
                logging.info('maitain_only all rest rate:{:.1f}%,correct:{}/{},miss{}'.format((1-num_rest/len(data_rest))*100,len(data_rest)-num_rest,len(data_rest),miss_rest))
                logging.info('maitain_only sampled rest rate:{:.1f}%,correct:{}/{},miss{}'.format((1-num_sampled/len(data_sampled))*100,len(data_sampled)-num_sampled,len(data_sampled),miss_sampled))
                logging.info('maitain_only sampled all rate:{:.1f}%,correct:{}/{},miss{}'.format((1-num_sampled_all/len(data_sampled_all))*100,len(data_sampled_all)-num_sampled_all,len(data_sampled_all),miss_sampled_all))
                logging.info('maitain_only knn_last rate:{:.1f}%,correct:{}/{},miss{}'.format((1-num_knn_last/len(data_knn_last))*100,len(data_knn_last)-num_knn_last,len(data_knn_last),miss_knn_last))
                logging.info('maitain_only knn_pass rate:{:.1f}%,correct:{}/{},miss{}'.format((1-num_knn_pass/len(data_knn_pass))*100,len(data_knn_pass)-num_knn_pass,len(data_knn_pass),miss_knn_pass))
                #存储maintain_only数据
                if filter_flag:
                    output_path=f"{root}/final_results/diff/maintain_only/filter/"
                    output_path_rest=f'{root}/final_results/diff/maintain_rest/filter/'
                    output_path_sampled=f'{root}/final_results/diff/maintain_sampled/filter/'
                    output_path_KNN=f'{root}/final_results/diff/maintain_knn_{method}/filter/'
                    output_path_low=f'{root}/final_results/diff/maintain_low/filter/'
                    output_path_sampled_all=f'{root}/final_results/diff/maintain_sampled_all/filter/'
                else:
                    output_path=f"{root}/final_results/diff/maintain_only/"
                    output_path_rest=f'{root}/final_results/diff/maintain_rest/'
                    output_path_sampled=f'{root}/final_results/diff/maintain_sampled/'
                    output_path_KNN=f'{root}/final_results/diff/maintain_knn_{method}/'
                    output_path_low=f'{root}/final_results/diff/maintain_low/'
                    output_path_sampled_all=f'{root}/final_results/diff/maintain_sampled_all/'

                os.makedirs(output_path+f'{mask_file}/', exist_ok=True)
                os.makedirs(output_path_rest+f'{mask_file}/', exist_ok=True)
                os.makedirs(output_path_sampled+f'{mask_file}/', exist_ok=True)
                os.makedirs(output_path_KNN+f'{mask_file}/', exist_ok=True)
                os.makedirs(output_path_low+f'{mask_file}/', exist_ok=True)
                os.makedirs(output_path_sampled_all+f'{mask_file}/', exist_ok=True)
                if mode=='att':
                    with open(output_path+f'{mask_file}/{model_name}-{data_name}-{sub_set}-{mode}-right-top_percentage{top_k_percentage}.json', 'w', encoding='utf-8') as f:
                            json.dump(diff, f, ensure_ascii=False, indent=4)
                    with open(output_path_rest+f'{mask_file}/{model_name}-{data_name}-{sub_set}-{mode}-right-top_percentage{top_k_percentage}.json', 'w', encoding='utf-8') as f:
                            json.dump(diff_rest, f, ensure_ascii=False, indent=4)
                    with open(output_path_sampled+f'{mask_file}/{model_name}-{data_name}-{sub_set}-{mode}-right-top_percentage{top_k_percentage}.json', 'w', encoding='utf-8') as f:
                            json.dump(diff_sampled, f, ensure_ascii=False, indent=4)
                    with open(output_path_KNN+f'{mask_file}/{model_name}-{data_name}-{sub_set}-knn-last-top_percentage{top_k_percentage}.json', 'w', encoding='utf-8') as f:
                            json.dump(diff_knn_last, f, ensure_ascii=False, indent=4)
                    with open(output_path_KNN+f'{mask_file}/{model_name}-{data_name}-{sub_set}-knn-pass-top_percentage{top_k_percentage}.json', 'w', encoding='utf-8') as f:
                            json.dump(diff_knn_pass, f, ensure_ascii=False, indent=4)
                    with open(output_path_low+f'{mask_file}/{model_name}-{data_name}-{sub_set}-{mode}-right-top_percentage{top_k_percentage}.json', 'w', encoding='utf-8') as f:
                            json.dump(diff_low, f, ensure_ascii=False, indent=4)
                    with open(output_path_sampled_all+f'{mask_file}/{model_name}-{data_name}-{sub_set}-{mode}-right-top_percentage{top_k_percentage}.json', 'w', encoding='utf-8') as f:
                            json.dump(diff_sampled_all, f, ensure_ascii=False, indent=4)
                elif mode=='grad':
                    with open(output_path+f'{mask_file}/{model_name}-{data_name}-{sub_set}-{mode}-correct-right-top_percentage{top_k_percentage}.json', 'w', encoding='utf-8') as f:
                            json.dump(diff, f, ensure_ascii=False, indent=4)
                    with open(output_path_rest+f'{mask_file}/{model_name}-{data_name}-{sub_set}-{mode}-correct-right-top_percentage{top_k_percentage}.json', 'w', encoding='utf-8') as f:
                            json.dump(diff_rest, f, ensure_ascii=False, indent=4)
                    with open(output_path_sampled+f'{mask_file}/{model_name}-{data_name}-{sub_set}-{mode}-correct-right-top_percentage{top_k_percentage}.json', 'w', encoding='utf-8') as f:
                            json.dump(diff_sampled, f, ensure_ascii=False, indent=4)
                else:
                    with open(output_path+f'{mask_file}/{model_name}-{data_name}-{sub_set}-{mode}-grad_correct-right-top_percentage{top_k_percentage}.json', 'w', encoding='utf-8') as f:
                            json.dump(diff, f, ensure_ascii=False, indent=4)
                    with open(output_path_rest+f'{mask_file}/{model_name}-{data_name}-{sub_set}-{mode}-grad_correct-right-top_percentage{top_k_percentage}.json', 'w', encoding='utf-8') as f:
                            json.dump(diff_rest, f, ensure_ascii=False, indent=4)
                    with open(output_path_sampled+f'{mask_file}/{model_name}-{data_name}-{sub_set}-{mode}-grad_correct-right-top_percentage{top_k_percentage}.json', 'w', encoding='utf-8') as f:
                            json.dump(diff_sampled, f, ensure_ascii=False, indent=4)
    #读取maintain_sentences数据
    for top_k_percentage in top_k_percentage_list:
        input_path_high=f'{root}/final_results/maintain_only_sentences/high_att/'
        input_path_low=f'{root}/final_results/maintain_only_sentences/low_att/'
        input_path_random=f'{root}/final_results/maintain_only_sentences/random/'
        input_path_random_wo_top_low=f'{root}/final_results/maintain_only_sentences/random_wo_top_low/'
        input_file=f"{model_name}-{data_name}-{sub_set}-att-right-top_percentage{top_k_percentage}.json"
        with open(input_path_high+input_file) as f:
                    data_high=json.load(f)
        with open(input_path_low+input_file) as f:
                    data_low=json.load(f)
        with open(input_path_random+input_file) as f:
                    data_random=json.load(f)
        with open(input_path_random_wo_top_low+input_file) as f:
                    data_random_wo_top_low=json.load(f)
        num_high,diff_high,miss_high=simple_analyze(data_high)
        num_low,diff_low,miss_low=simple_analyze(data_low)
        num_random,diff_random,miss_random=simple_analyze(data_random)  
        num_random_wo_top_low,diff_random_wo_top_low,miss_random_wo_top_low=simple_analyze(data_random_wo_top_low)  
        logging.info("="*10+f"Maintain Sentences top_k_percentage={top_k_percentage}"+'='*10)
        logging.info('maitain sentences high rate:{:.1f}%,correct:{}/{},miss{}'.format((1-num_high/len(data_high))*100,len(data_high)-num_high,len(data_high),miss_high))
        logging.info('maitain sentences low rate:{:.1f}%,correct:{}/{},miss{}'.format((1-num_low/len(data_low))*100,len(data_low)-num_low,len(data_low),miss_low))
        logging.info('maitain sentences random rate:{:.1f}%,correct:{}/{},miss{}'.format((1-num_random/len(data_random))*100,len(data_random)-num_random,len(data_random),miss_random))
        logging.info('maitain sentences random_wo_top_low rate:{:.1f}%,correct:{}/{},miss{}'.format((1-num_random_wo_top_low/len(data_random_wo_top_low))*100,len(data_random_wo_top_low)-num_random_wo_top_low,len(data_random_wo_top_low),miss_random_wo_top_low))                    
        output_path_high=f"{root}/final_results/diff/maintain_only_sentences/high_att/"
        output_path_low=f'{root}/final_results/diff/maintain_only_sentences/low_att/'
        output_path_random=f'{root}/final_results/diff/maintain_only_sentences/random/'
        output_path_random_wo_top_low=f'{root}/final_results/diff/maintain_only_sentences/random_wo_top_low/'
        os.makedirs(output_path_high, exist_ok=True)
        os.makedirs(output_path_low, exist_ok=True)
        os.makedirs(output_path_random, exist_ok=True)
        os.makedirs(output_path_random_wo_top_low, exist_ok=True)
        output_file=f'{model_name}-{data_name}-{sub_set}-att-right-top_percentage{top_k_percentage}_diff.json'
        with open(output_path_high+output_file, 'w', encoding='utf-8') as f:
            json.dump(diff_high, f, ensure_ascii=False, indent=4)
        with open(output_path_low+output_file, 'w', encoding='utf-8') as f:
            json.dump(diff_low, f, ensure_ascii=False, indent=4)
        with open(output_path_random+output_file, 'w', encoding='utf-8') as f:
            json.dump(diff_random, f, ensure_ascii=False, indent=4)
        with open(output_path_random+output_file, 'w', encoding='utf-8') as f:
            json.dump(diff_random_wo_top_low, f, ensure_ascii=False, indent=4)
    input_path_model_select=f'{root}/final_results/maintain_only_sentences/model_select/'
    input_file=f"{model_name}-{data_name}-{sub_set}-model_select.json"
    with open(input_path_model_select+input_file) as f:
            data_model_select=json.load(f)
    num_model_select,diff_model_select,miss_model_select=simple_analyze(data_model_select)
    logging.info("="*10+f"Maintain Selected Sentences"+'='*10)
    logging.info('maitain selected sentences rate:{:.1f}%,correct:{}/{},miss{}'.format((1-num_model_select/len(data_model_select))*100,len(data_model_select)-num_model_select,len(data_model_select),miss_model_select))
    output_path_model_select=f'{root}/final_results/diff/maintain_only_sentences/model_select/'
    os.makedirs(output_path_model_select, exist_ok=True)
    output_file_model_select=f'{model_name}-{data_name}-{sub_set}-model_select_diff.json'
    with open(output_path_model_select+output_file_model_select, 'w', encoding='utf-8') as f:
        json.dump(diff_model_select, f, ensure_ascii=False, indent=4)
    logging.info(f"finish analyze for {data_name}_{sub_set}_{model_name}")
