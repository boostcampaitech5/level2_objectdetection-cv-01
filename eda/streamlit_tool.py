import streamlit as st
from simple_loader import *
from eda_fns import *
import eda_fns
import matplotlib.pyplot as plt

#streamlit run streamlit_tool.py --server.port 30003 --server.fileWatcherType none
import eda_fns
@st.cache_resource
class DataClass():
    def __init__(self):
        print('make dataclass')
        self.num_class_plt = eda_class_disribution()
        self.object_per_image_plt = eda_one_image_object_distribution()
        self.total_num = len(eda_fns.data_df)
        self.class_per_image_plt = eda_one_image_class_distribution()
        self.box_pos_plt = eda_box_position_distribution()
        self.box_area_plt = eda_box_area_distribution()
        self.box_ratio_plt = eda_box_wh_ratio_distribution()
        

data_class = DataClass()
eda_type = st.sidebar.selectbox('EDA type',('default','이미지 보기','bbox이미지보기'
                                            ,'데이터셋 분류'
                                            ,'클래스 분포', '이미지당 오브젝트 개수'
                                            ,'이미지당 클래스 개수', '박스의 중심점 분포'
                                            ,'박스의 크기 분포','박스의 가로세로 비율 분포'
                                            ,'target class 바꾸기'
                                            ))

if eda_type == 'target class 바꾸기':
    st.text('class 종류: 0.General trash, 1.Paper, 2.Paper pack, 3.Metal, 4.Glass, 5.Plastic, 6.Styrofoam, 7.Plastic bag, 8.Battery, 9.Clothing')
    switch_class = st.selectbox('변경할 class를 선택하세요.',['all',0,1,2,3,4,5,6,7,8,9])

    if switch_class in ['all',0,1,2,3,4,5,6,7,8,9]:
        eda_fns.data_df, eda_fns.ann_df = json_to_df()
        if switch_class in [0,1,2,3,4,5,6,7,8,9]:
            st.text(f'switch class to {switch_class}')
            eda_fns.ann_df = eda_fns.ann_df[eda_fns.ann_df['category_id']==switch_class]    
            eda_fns.data_df = eda_fns.data_df.iloc[eda_fns.ann_df['image_id'].unique()]
            
        
        data_class.num_class_plt = eda_class_disribution()
        data_class.object_per_image_plt = eda_one_image_object_distribution()
        data_class.total_num = len(eda_fns.data_df)
        data_class.class_per_image_plt = eda_one_image_class_distribution()
        data_class.box_pos_plt = eda_box_position_distribution()
        data_class.box_area_plt = eda_box_area_distribution()
        data_class.box_ratio_plt = eda_box_wh_ratio_distribution()

if eda_type == '이미지 보기':
    if 'image_index' not in st.session_state:
        st.session_state.image_index = 0
    col1,col2,col3,col4 = st.columns([1,1,3,1])
    prev_btn = col1.button('prev')
    next_btn = col2.button('next')
    target_idx = col3.text_input('move index')
    index_move = col4.button('move')
    if prev_btn:
        st.session_state.image_index = (st.session_state.image_index -1)%data_class.total_num
    if next_btn:
        st.session_state.image_index = (st.session_state.image_index +1)%data_class.total_num
    if index_move:
        target_idx = int(target_idx)
        if target_idx>=0 and target_idx< data_class.total_num:
            st.session_state.image_index = target_idx

    target_img = eda_fns.data_df.iloc[st.session_state.image_index]
    print(eda_fns.data_df[:5])
    print(target_img)
    st.image(get_img(target_img))
    st.text(f'Img idx: {st.session_state.image_index }/{data_class.total_num}\n\
            info\n{target_img}')
    
if eda_type == 'bbox이미지보기':
    if 'image_index' not in st.session_state:
        st.session_state.image_index = 0
    col1,col2,col3,col4 = st.columns([1,1,4,1])
    prev_btn = col1.button('prev')
    next_btn = col2.button('next')
    target_idx = col3.text_input('move index')
    index_move = col4.button('move')
    if prev_btn:
        st.session_state.image_index = (st.session_state.image_index -1)%data_class.total_num
    if next_btn:
        st.session_state.image_index = (st.session_state.image_index +1)%data_class.total_num
    if index_move:
        target_idx = int(target_idx)
        if target_idx>=0 and target_idx< data_class.total_num:
            st.session_state.image_index = target_idx
            
    target_img = eda_fns.data_df.iloc[st.session_state.image_index]

    anns = eda_fns.ann_df[eda_fns.ann_df['image_id']==target_img['id']]
    vis_anns = [True]*len(anns)
    check_list = [i for i in range(len(anns))]
    checks = st.multiselect("안보이게 할 bbox를 체크하세요",check_list)
    for c in checks:
        vis_anns[c]=False
    st.pyplot(get_img_with_bbox(target_img, anns,vis_anns))
    st.text(f'Img idx: {st.session_state.image_index }/{data_class.total_num}\n\
            info\n{target_img}')
    st.text(f'Ann info\n{anns}')

if eda_type == '데이터셋 분류':
    if 'classify_idx' not in st.session_state:
        st.session_state.classify_idx = 0
        st.session_state.high_ann = []
        st.session_state.mid_ann = []
        st.session_state.low_ann = []
    st.text('high: 바로 사용하기 좋음. mid: annotation 약간 수정필요. low: 데이터셋에서 제거가 좋음.')
    reset_btn = st.button('reset')
    if reset_btn:
        st.session_state.high_ann = []
        st.session_state.mid_ann = []
        st.session_state.low_ann = []
        st.session_state.classify_idx = 0
    if st.session_state.classify_idx < data_class.total_num:
        target_img = eda_fns.data_df.iloc[st.session_state.classify_idx]
        col1,col2,col3 = st.columns([1,1,1])
        with col1:
            high = st.button('high')
        with col2:
            mid = st.button('mid')
        with col3:
            low = st.button('low')
        if high:
            st.session_state.high_ann.append(target_img.to_dict())
            st.session_state.classify_idx+=1
        elif mid:
            st.session_state.mid_ann.append(target_img.to_dict())
            st.session_state.classify_idx+=1
        elif low:
            st.session_state.low_ann.append(target_img.to_dict())
            st.session_state.classify_idx+=1
        if st.session_state.classify_idx < data_class.total_num:
            target_img = eda_fns.data_df.iloc[st.session_state.classify_idx]
            st.text(f'Idx: {st.session_state.classify_idx}:{data_class.total_num}')
            anns = eda_fns.ann_df[eda_fns.ann_df['image_id']==target_img['id']]
            st.pyplot(get_img_with_bbox(target_img, anns))
        else:
            st.button('end')
    else: 
        st.text(f"high: {len(st.session_state.high_ann)}개\nmid:{len(st.session_state.mid_ann)}개\nlow:{len(st.session_state.low_ann)}개")
        st.text("저장하시겠습니까?")
        save_btn = st.button('save')
        if save_btn:
            print(st.session_state.high_ann)
            save_classify_json(st.session_state.high_ann,st.session_state.mid_ann,st.session_state.low_ann)

if eda_type == '클래스 분포':
    st.pyplot(data_class.num_class_plt[0])
    st.dataframe(data_class.num_class_plt[2])
if eda_type == '이미지당 오브젝트 개수':
    st.pyplot(data_class.object_per_image_plt[0])
    st.dataframe(data_class.object_per_image_plt[2])
if eda_type == '이미지당 클래스 개수':
    st.pyplot(data_class.class_per_image_plt[0])
if eda_type == '박스의 중심점 분포':
    st.pyplot(data_class.box_pos_plt[0])
if eda_type == '박스의 크기 분포':
    st.pyplot(data_class.box_area_plt[0])
if eda_type == '박스의 가로세로 비율 분포':
    st.pyplot(data_class.box_ratio_plt[0])