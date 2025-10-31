import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import FixedLocator
import mpl_finance  # 这是一个常用的 K 线图工具库
from io import BytesIO
import base64
from typing import List, Tuple
from pydantic import BaseModel, Field
from fastapi import FastAPI
import uvicorn

# Pydantic 数据模型 (如上所示)
KLine = Tuple[str, float, float, float, float, float] # (timestamp, open, high, low, close, volume_quote)
HLine = Tuple[float, float, float, float] # (open_price, take_profit, stop_loss, confidence)

class ChartRequest(BaseModel):
    klines: str
    hlines: str
    coin: str


def extract_klines_from_str(text: str) -> List[KLine]:
    """
    从字符串中提取 K 线数据。
    
    Args:
        text: CSV格式字符串，包含header行和数据行
        
    Returns:
        List[KLine]: K线数据列表，格式为 (timestamp, open, high, low, close, volume_quote)
    """
    lines = text.strip().split('\n')
    if len(lines) < 2:
        return []
    
    # 跳过header行（第一行）
    klines = []
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        
        # 分割CSV行，处理可能的空格
        parts = [p.strip() for p in line.split(',')]
        if len(parts) < 6:
            continue
        
        try:
            timestamp = parts[0]  # 保持为字符串
            open_price = float(parts[1])
            high_price = float(parts[2])
            low_price = float(parts[3])
            close_price = float(parts[4])
            volume_quote = float(parts[5])
            
            klines.append((timestamp, open_price, high_price, low_price, close_price, volume_quote))
        except (ValueError, IndexError) as e:
            # 跳过无法解析的行
            continue
    
    return klines


def extract_hlines_from_str(text: str) -> List[HLine]:
    """
    从字符串中提取交易建议数据（HLine）。
    
    Args:
        text: Markdown表格格式字符串，包含header行、分隔行和数据行
        
    Returns:
        List[HLine]: 交易建议列表，格式为 (open_price, take_profit, stop_loss, confidence)
    """
    lines = text.strip().split('\n')
    if len(lines) < 3:
        return []
    
    # 跳过header行（第一行）和分隔行（第二行，通常是 "| --- | ..."）
    hlines = []
    for line in lines[2:]:
        line = line.strip()
        if not line or line.startswith('|---'):
            continue
        
        # 分割markdown表格行
        # 格式: | 策略 | 方向 | 置信度 | 入场价 | 止盈 | 止损 | 注释 |
        parts = [p.strip() for p in line.split('|')]
        # 移除首尾空元素（因为行首和行尾的|会产生空字符串）
        parts = [p for p in parts if p]
        
        if len(parts) < 6:
            continue
        
        try:
            # 索引对应: 策略(0), 方向(1), 置信度(2), 入场价(3), 止盈(4), 止损(5), 注释(6...)
            confidence = float(parts[2])
            entry_price = float(parts[3])  # 入场价
            take_profit = float(parts[4])  # 止盈
            stop_loss = float(parts[5])    # 止损
            
            hlines.append((entry_price, take_profit, stop_loss, confidence))
        except (ValueError, IndexError) as e:
            # 跳过无法解析的行
            continue
    # 打乱 hlines （数组长度为8） 的顺序，按 0,4,1,5,2,6,3,7 排列
    index_map = [0, 4, 1, 5, 2, 6, 3, 7]
    hlines = [hlines[i] for i in index_map]
    return hlines


app = FastAPI(
    title="K-Line Chart Microservice",
    description="Generate K-Line charts with profit/loss areas and trading suggestions."
)

def generate_chart_base64_from_text(text_klines: str, text_hlines: str, coin: str) -> str:
    klines = extract_klines_from_str(text_klines)[-40:]
    hlines = extract_hlines_from_str(text_hlines)
    return generate_chart_base64(klines, hlines, coin)

def generate_chart_base64(klines: List[KLine], hlines: List[HLine], coin: str) -> str:
    """
    生成 K 线图并叠加交易建议区域，返回 base64 编码的 PNG 字符串。
    """
    # 1. 数据准备
    if not klines:
        raise ValueError("No klines data provided")
    
    df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume_quote'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    num_klines = len(df)
    
    # 转换为 Matplotlib OHLC 格式: 使用整数索引作为 x 坐标 (0, 1, 2, ...)
    # 这样 candlestick 和 hlines 使用相同的坐标系统
    df['index'] = range(num_klines)
    ohlc = df[['index', 'open', 'high', 'low', 'close']].values

    # 2. 创建画布
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # 3. 绘制 K 线 (忽略成交量)
    mpl_finance.candlestick_ohlc(
        ax, ohlc, width=0.8, colorup='red', colordown='green', alpha=1.0
    )
    
    # 4. 绘制交易建议区域
    suggestion_width = 4.0 # 建议区域宽度为 4 根 K 线
    
    # 从图表最右边（最后一根 K 线的 X 坐标）开始向左绘制
    start_x = num_klines - 1
    
    for i, (open_price, take_profit, stop_loss, confidence) in enumerate(hlines):
        # 计算 X 轴起始位置。每绘制一个建议，就向左移动 suggestion_width
        # X 轴坐标是 K 线的索引，从 0 开始
        rect_start_x = start_x - (i + 1) * suggestion_width + 0.5  # +0.5 微调居中
        rect_end_x = rect_start_x + suggestion_width

        # 止盈区域 (绿色, alpha=confidence^2)
        alpha_value = min(1.0, confidence ** 2)  # 限制alpha在0-1之间
        tp_rect_height = abs(take_profit - open_price)
        tp_rect_y = min(open_price, take_profit)
        ax.add_patch(
            Rectangle(
                (rect_start_x, tp_rect_y), 
                suggestion_width, 
                tp_rect_height, 
                facecolor='green', 
                alpha=alpha_value, 
                zorder=1 # 确保在 K 线下方或同层
            )
        )

        # 止损区域 (红色, alpha=confidence^2)
        sl_rect_height = abs(stop_loss - open_price)
        sl_rect_y = min(open_price, stop_loss)
        ax.add_patch(
            Rectangle(
                (rect_start_x, sl_rect_y), 
                suggestion_width, 
                sl_rect_height, 
                facecolor='red', 
                alpha=alpha_value,
                zorder=1
            )
        )
        
        # # 绘制开仓价细线 (蓝色)
        # if num_klines > 1:
        #     xmin_frac = rect_start_x / (num_klines - 1)
        #     xmax_frac = rect_end_x / (num_klines - 1)
        # else:
        #     xmin_frac = 0
        #     xmax_frac = 1
        # ax.axhline(
        #     y=open_price, 
        #     xmin=xmin_frac, 
        #     xmax=xmax_frac, 
        #     color='blue', 
        #     linewidth=1, 
        #     linestyle='-', 
        #     zorder=2
        # )
        
        # 标注交易建议和置信度
        text_x = rect_start_x + suggestion_width / 2 # 居中
        ax.update_datalim([[rect_start_x, min(take_profit, stop_loss, open_price)], 
                          [rect_end_x, max(take_profit, stop_loss, open_price)]])
        ax.autoscale_view()
        ylim = ax.get_ylim()
        label_confidence = f"Long Conf: {confidence:.2f}" if take_profit > open_price else f"Short Conf: {confidence:.2f}"
        if take_profit > open_price:
            text_y = max(take_profit, stop_loss, open_price) + (ylim[1] - ylim[0]) * 0.02
            ax.text(
                text_x, text_y, 
                f"Entry: {int(open_price)}\n{int(take_profit)} / {int(stop_loss)}\n{label_confidence}",
                ha='center', va='bottom', fontsize=11, color='black', bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 3}
            )
        else:
            text_y = min(take_profit, stop_loss, open_price) - (ylim[1] - ylim[0]) * 0.02
            ax.text(
                text_x, text_y, 
                f"Entry: {int(open_price)}\n{int(take_profit)} / {int(stop_loss)}\n{label_confidence}",
                ha='center', va='top', fontsize=11, color='black', bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 3}
            )


    # 5. 图表美化
    ax.autoscale_view()
    # Set up FixedLocator and FixedFormatter together to avoid warnings
    # 使用整数索引作为 tick 位置
    tick_positions = df['index'].values
    tick_labels = df['timestamp'].dt.strftime('%H:%M').values
    
    # 如果 K 线太多，只显示部分 ticks
    if num_klines > 20:
        step = max(1, num_klines // 20)
        tick_positions = tick_positions[::step]
        tick_labels = tick_labels[::step]
    
    ax.xaxis.set_major_locator(FixedLocator(tick_positions))
    ax.xaxis.set_major_formatter(plt.FixedFormatter(tick_labels))
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_title(f"{coin} Multi Targets", fontsize=16)
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Price", fontsize=12)
    
    # 在右上角标注最后一根K线的日期
    last_kline_date = df['timestamp'].iloc[-1]
    date_str = last_kline_date.strftime('%Y-%m-%d %H:%M')
    ax.text(
        0.98, 0.98, 
        date_str, 
        transform=ax.transAxes,
        ha='right', va='top',
        fontsize=12, 
        color='black',
        bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 5}
    )
    
    plt.tight_layout()

    # 6. Base64 编码
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=50, bbox_inches='tight')
    plt.close(fig)
    data = base64.b64encode(buf.getbuffer()).decode("utf-8")
    
    return data

@app.post("/generate-chart")
async def generate_chart(request: ChartRequest):
    """
    根据 K 线数据和交易建议生成包含盈亏区域的 K 线图 (PNG base64 编码)。
    """
    try:
        chart_base64 = generate_chart_base64_from_text(request.klines, request.hlines, request.coin)
        return chart_base64
    except Exception as e:
        # 生产环境中应该记录更详细的错误日志
        return {"error": f"Failed to generate chart: {e}"}, 500

# 运行微服务 (用于本地测试)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9012)
    pass