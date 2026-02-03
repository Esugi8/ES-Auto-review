import streamlit as st
import google.generativeai as genai
import pandas as pd
import io
import json

# --- 設定 ---
st.set_page_config(page_title="AI面接官：ES分析くん", layout="wide", page_icon="📄")

# カスタムCSSでデザインを少し整える
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #FF4B4B; color: white; }
    </style>
    """, unsafe_allow_html=True)

st.title("📄 AI面接質問生成ツール (Advanced)")
st.caption("エントリーシートを分析し、構造化面接のための質問リストを自動生成します。")

# --- セッション状態の初期化 ---
if 'df_questions' not in st.session_state:
    st.session_state.df_questions = None

# APIキー設定
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    model = genai.GenerativeModel('gemini-flash-latest')
else:
    st.error("APIキーが設定されていません。")
    st.stop()

# --- サイドバー ---
with st.sidebar:
    st.header("⚙️ 設定")
    uploaded_file = st.file_uploader("ES(PDF)をアップロード", type="pdf")
    
    generation_config = {
        "temperature": 0.7,
        "top_p": 0.95,
        "response_mime_type": "application/json",
    }
    
    st.info("※1.5 Flashモデルを使用しています。")

# --- メイン処理 ---
if uploaded_file:
    if st.button("✨ 質問リストを生成する"):
        with st.spinner("AIが深く読み込んでいます..."):
            try:
                pdf_data = uploaded_file.read()

                # プロンプトの改良（JSON出力を強制）
                prompt = """
                添付されたエントリーシートを読み取り、面接官用の質問リストを作成してください。
                出力は必ず以下のJSON形式のリストで返してください。
                
                [
                  {
                    "セクション": "学業・ゼミ・研究",
                    "メイン質問": "...",
                    "深掘り質問": "...",
                    "評価の着眼点": "..."
                  },
                  ...
                ]

                【制約事項】
                1. セクションは必ず以下の5つに分類し、各3問ずつ作成してください：
                   「学業・ゼミ・研究」「学業以外（インターン）」「周囲を巻き込んだ経験」「志望動機」「5年後の姿」
                2. 評価の着眼点は、具体的かつ客観的な指標を提示してください。
                """

                # Geminiにリクエスト
                response = model.generate_content(
                    [prompt, {"mime_type": "application/pdf", "data": pdf_data}],
                    generation_config=generation_config
                )

                # JSONパース
                res_json = json.loads(response.text)
                df = pd.DataFrame(res_json)
                
                # 空の列を追加
                df["回答メモ"] = ""
                df["評価(1-5)"] = ""
                
                # セッションに保存
                st.session_state.df_questions = df

            except Exception as e:
                st.error(f"エラーが発生しました: {e}")

# --- 結果表示 ---
if st.session_state.df_questions is not None:
    df = st.session_state.df_questions
    
    st.success("分析が完了しました！")
    
    # タブで表示を分ける
    tab1, tab2 = st.tabs(["📋 質問リスト表示", "📥 ダウンロード"])
    
    with tab1:
        st.subheader("生成された質問リスト")
        # 編集可能なデータフレーム
        edited_df = st.data_editor(df, use_container_width=True, num_rows="fixed")
    
    with tab2:
        col1, col2 = st.columns(2)
        
        # CSVダウンロード
        csv_buffer = io.StringIO()
        edited_df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
        col1.download_button(
            label="CSVをダウンロード",
            data=csv_buffer.getvalue(),
            file_name="interview_sheet.csv",
            mime="text/csv"
        )
        
        # Excelダウンロード用 (要: openpyxl)
        try:
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                edited_df.to_excel(writer, index=False, sheet_name='質問リスト')
            col2.download_button(
                label="Excelファイルをダウンロード",
                data=excel_buffer.getvalue(),
                file_name="interview_sheet.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        except ImportError:
            col2.warning("Excel出力には openpyxl が必要です。")

else:
    st.info("PDFをアップロードして「生成」ボタンを押してください。")

# --- フッター ---
st.markdown("---")
st.caption("Powered by Google Gemini 1.5 Flash & Streamlit")