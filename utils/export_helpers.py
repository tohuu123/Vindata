"""
export_helpers.py
-----------------
Các hàm tiện ích để xuất kết quả phân tích (biểu đồ và bảng dữ liệu).

Cách dùng:
----------
    from utils.export_helpers import save_plot, save_dataframe, save_results

    # Lưu figure matplotlib hiện tại vào thư mục con mặc định
    save_plot("ten_bieu_do")

    # Lưu figure matplotlib vào thư mục con tùy chọn
    save_plot("ten_bieu_do", subfolder="products")

    # Lưu DataFrame thành CSV
    save_dataframe(df, "ten_bang")
    save_dataframe(df, "ten_bang", subfolder="orders")

    # Lưu cả figure lẫn DataFrame trong một lần gọi
    save_results(fig=plt.gcf(), df=df, name="phan_tich_san_pham", subfolder="products")

Cấu trúc thư mục được tạo ra:
------------------------------
    <root>/results/                 ← thư mục gốc mặc định
        <subfolder>/
            plots/                  ← ảnh PNG
            tables/                 ← file CSV

Tham số chung:
--------------
    name      : Tên file (không cần phần mở rộng).
    subfolder : Tên thư mục con bên trong thư mục gốc.
                Mặc định là chuỗi rỗng "" (lưu thẳng vào <root>).
    root_dir  : Thư mục gốc. Mặc định là "results" (tương đối so với
                thư mục làm việc hiện tại).
    fmt       : Định dạng hình ảnh (mặc định "png"). Dùng cho save_plot /
                save_results.
    dpi       : Độ phân giải hình ảnh (mặc định 150). Dùng cho save_plot /
                save_results.
    encoding  : Mã hóa CSV (mặc định "utf-8-sig" để Excel hiển thị
                tiếng Việt đúng).
    index     : Có ghi index của DataFrame vào CSV không (mặc định True).
"""

import os
import matplotlib.pyplot as plt
import pandas as pd


# ---------------------------------------------------------------------------
# Hàm nội bộ
# ---------------------------------------------------------------------------

def _make_dir(path: str) -> None:
    """Tạo thư mục (bao gồm các thư mục cha) nếu chưa tồn tại."""
    os.makedirs(path, exist_ok=True)


def _build_path(
    category: str,
    name: str,
    ext: str,
    subfolder: str = "",
    root_dir: str = "results",
) -> str:
    """
    Xây dựng đường dẫn đầy đủ để lưu file.

    Parameters
    ----------
    category : str
        Loại dữ liệu: "plots" hoặc "tables".
    name : str
        Tên file (không có phần mở rộng).
    ext : str
        Phần mở rộng file (ví dụ "png", "csv").
    subfolder : str, optional
        Tên thư mục con bên trong root_dir. Mặc định là "" (không có
        thư mục con, lưu thẳng vào root_dir/<category>/).
    root_dir : str, optional
        Thư mục gốc. Mặc định là "results".

    Returns
    -------
    str
        Đường dẫn file đã được tạo sẵn thư mục.
    """
    if subfolder:
        dir_path = os.path.join(root_dir, subfolder, category)
    else:
        dir_path = os.path.join(root_dir, category)

    _make_dir(dir_path)
    return os.path.join(dir_path, f"{name}.{ext}")


# ---------------------------------------------------------------------------
# Hàm công khai
# ---------------------------------------------------------------------------

def save_plot(
    name: str,
    fig=None,
    subfolder: str = "",
    root_dir: str = "results",
    fmt: str = "png",
    dpi: int = 150,
    bbox_inches: str = "tight",
) -> str:
    """
    Lưu một figure matplotlib thành file ảnh.

    Parameters
    ----------
    name : str
        Tên file (không cần phần mở rộng).
    fig : matplotlib.figure.Figure, optional
        Figure cần lưu. Nếu None thì dùng figure hiện tại (plt.gcf()).
    subfolder : str, optional
        Thư mục con bên trong root_dir. Mặc định "" (không có thư mục con).
    root_dir : str, optional
        Thư mục gốc. Mặc định "results".
    fmt : str, optional
        Định dạng ảnh (mặc định "png").
    dpi : int, optional
        Độ phân giải (mặc định 150).
    bbox_inches : str, optional
        Cắt lề ảnh (mặc định "tight").

    Returns
    -------
    str
        Đường dẫn file đã lưu.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> plt.plot([1, 2, 3])
    >>> save_plot("line_chart")                        # → results/plots/line_chart.png
    >>> save_plot("line_chart", subfolder="products")  # → results/products/plots/line_chart.png
    """
    if fig is None:
        fig = plt.gcf()

    file_path = _build_path("plots", name, fmt, subfolder=subfolder, root_dir=root_dir)
    fig.savefig(file_path, dpi=dpi, bbox_inches=bbox_inches)
    print(f"[export_helpers] Đã lưu biểu đồ: {file_path}")
    return file_path


def save_dataframe(
    df: pd.DataFrame,
    name: str,
    subfolder: str = "",
    root_dir: str = "results",
    encoding: str = "utf-8-sig",
    index: bool = True,
) -> str:
    """
    Lưu một DataFrame thành file CSV.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame cần lưu.
    name : str
        Tên file (không cần phần mở rộng).
    subfolder : str, optional
        Thư mục con bên trong root_dir. Mặc định "" (không có thư mục con).
    root_dir : str, optional
        Thư mục gốc. Mặc định "results".
    encoding : str, optional
        Mã hóa CSV (mặc định "utf-8-sig" để Excel hiển thị tiếng Việt đúng).
    index : bool, optional
        Có ghi index của DataFrame vào CSV không (mặc định True).

    Returns
    -------
    str
        Đường dẫn file đã lưu.

    Examples
    --------
    >>> save_dataframe(df_products, "products_stats")
    >>> save_dataframe(df_orders, "orders_summary", subfolder="orders", index=False)
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Tham số 'df' phải là pd.DataFrame, nhận được {type(df)}")

    file_path = _build_path("tables", name, "csv", subfolder=subfolder, root_dir=root_dir)
    df.to_csv(file_path, encoding=encoding, index=index)
    print(f"[export_helpers] Đã lưu bảng dữ liệu: {file_path}")
    return file_path


def save_results(
    name: str,
    fig=None,
    df: pd.DataFrame = None,
    subfolder: str = "",
    root_dir: str = "results",
    fmt: str = "png",
    dpi: int = 150,
    encoding: str = "utf-8-sig",
    index: bool = True,
) -> dict:
    """
    Tiện ích gộp: lưu biểu đồ VÀ/HOẶC DataFrame trong một lần gọi.

    Parameters
    ----------
    name : str
        Tên file chung (không cần phần mở rộng).
    fig : matplotlib.figure.Figure, optional
        Figure cần lưu. Nếu None và df cũng None thì không làm gì cả.
        Nếu fig là True hoặc không truyền nhưng df khác None, hàm sẽ
        thử lưu figure hiện tại (plt.gcf()) khi fig=True.
        Để bỏ qua việc lưu biểu đồ, truyền fig=None.
    df : pd.DataFrame, optional
        DataFrame cần lưu. Nếu None thì bỏ qua.
    subfolder : str, optional
        Thư mục con bên trong root_dir. Mặc định "" (không có thư mục con).
    root_dir : str, optional
        Thư mục gốc. Mặc định "results".
    fmt : str, optional
        Định dạng ảnh (mặc định "png").
    dpi : int, optional
        Độ phân giải ảnh (mặc định 150).
    encoding : str, optional
        Mã hóa CSV (mặc định "utf-8-sig").
    index : bool, optional
        Có ghi index DataFrame vào CSV không (mặc định True).

    Returns
    -------
    dict
        Dictionary với các key "plot" và "table" chứa đường dẫn file
        tương ứng (hoặc None nếu không lưu).

    Examples
    --------
    >>> # Lưu cả biểu đồ hiện tại và DataFrame
    >>> save_results("margin_analysis", fig=plt.gcf(), df=df_stats, subfolder="products")

    >>> # Chỉ lưu biểu đồ
    >>> save_results("revenue_chart", fig=plt.gcf())

    >>> # Chỉ lưu DataFrame
    >>> save_results("summary_table", df=df_summary, subfolder="orders")
    """
    paths = {"plot": None, "table": None}

    if fig is not None:
        paths["plot"] = save_plot(
            name,
            fig=fig,
            subfolder=subfolder,
            root_dir=root_dir,
            fmt=fmt,
            dpi=dpi,
        )

    if df is not None:
        paths["table"] = save_dataframe(
            df,
            name,
            subfolder=subfolder,
            root_dir=root_dir,
            encoding=encoding,
            index=index,
        )

    return paths
