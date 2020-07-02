function [dist,ix,iy] = dtw(x,y,varargin)
%DTW    Distance between signals via Dynamic Time Warping
%   DIST = DTW(X,Y) computes the total distance, DIST, as the minimum sum
%   of Euclidean distances between the samples of vectors X and Y, where
%   samples in either X or Y may repeat consecutively any number of times.
%   If X and Y are matrices, then X and Y must have the same number of rows
%   and DTW minimizes the total Euclidean distances between the column
%   vectors of X and Y, allowing columns of X and Y to consecutively
%   repeat.
%
%   [DIST,IX,IY] = DTW(X,Y) additionally returns the warping path, IX and
%   IY, that minimizes the total Euclidean distance between X(IX) and Y(IY)
%   when X and Y are vectors and between X(:,IX) and Y(:,IY) when X and Y
%   are matrices.
%
%   [DIST,IX,IY] = DTW(X,Y,MAXSAMP) additionally restricts IX and IY so
%   that they must be within MAXSAMP samples of a straight-line fit between
%   X and Y.  If MAXSAMP is unspecified, then no restriction will be placed
%   upon IX or IY.
%
%   [DIST,IX,IY] = DTW(X,Y,...,METRIC) will return in DIST the summed
%   distance between the corresponding entries of X and Y according to the
%   distance metric defined by METRIC.  The default is 'euclidean'.
%      'absolute'  the sum of the absolute (manhattan) differences
%      'euclidean' the root sum squared differences
%      'squared'   the squared Euclidean distance
%      'symmkl'    the symmetric Kullback-Leibler distance
%   When data and signal are matrices, the distances are taken between
%   corresponding column vectors; when data and signal are vectors,
%   distances are taken between the corresponding elements.  Note that when
%   data and signal are vectors, 'absolute' and 'euclidean' are equivalent.
%
%   DTW(...) without output arguments plots the original and aligned
%   signals.  DTW displays the alignment of X and Y via a line plot when
%   they are vectors, and via horizontally aligned images when they are
%   matrices.  If the matrices are complex, the real and imaginary
%   portions appear in the top and bottom half of each image, respectively.
%
%   % Example 1:
%   %   Compute and plot the best Euclidean distance between real
%   %   chirp and sinusoidal signals using dynamic time warping.
%   x = chirp(0:999,0,1000,1/100);
%   y = cos(2*pi*5*(0:199)/200);
%   dtw(x,y)
%
%   % Example 2:
%   %   Compute and plot the best Euclidean distance between complex
%   %   chirp and sinusoidal signals using dynamic time warping.
%   x = exp(2i*pi*(3*(1:1000)/1000).^2);
%   y = exp(2i*pi*9*(1:399)/400);
%   dtw(x,y)
%
%   % Example 3:
%   %   Align handwriting samples along the x-axis.
%   load blockletterex
%   dtw(MATLAB1,MATLAB2);
%
%   See also EDR, ALIGNSIGNALS, FINDSIGNAL, FINDDELAY, XCORR.

%   References:
%   * H. Sakoe and S. Chiba, "Dynamic Programming Algorithm Optimization
%     for Spoken Word Recognition" IEEE Transactions on Acoustics, Speech
%     and Signal Processing, Vol. ASSP-26, No. 1, Feb 1978, pp. 43-49.
%   * K.K. Paliwal, A. Agarwal and S.S. Sinha, "A Modification over Sakoe
%     and Chiba's dynamic time warping algorithm for isolated word
%     recognition" IEEE International Conference on ICASSP 1982., Vol. 7,
%     pp. 1259-1261

%   Copyright 2015-2019 The MathWorks, Inc.
%#codegen

narginchk(2,4);
validateattributes(x,{'single','double'},{'nonnan','2d','finite'},'dtw','X',1);
validateattributes(y,{'single','double'},{'nonnan','2d','finite'},'dtw','Y',2);

if nargin == 4
    if isnumeric(varargin{1})
        r = varargin{1};
        opt = varargin{2};
    elseif isnumeric(varargin{2})
        r = varargin{2};
        opt = varargin{1};
    else
        coder.internal.error('signal:getmutexclopt:ConflictingOptions');
    end
elseif nargin == 3
    if isnumeric(varargin{1})
        r = varargin{1};
        opt = 'euclidean';
    else
        opt = varargin{1};
        r = Inf;
    end
else
    opt = 'euclidean';
    r = Inf;
end

validopts = {'absolute','euclidean','squared','symmkl'};
metric = validatestring(opt,validopts,'dtw');

needsTranspose = iscolumn(x);
if iscolumn(x) && isvector(y)
    xSignal = x.';
else
    xSignal = x;
end

if iscolumn(y) && isvector(x)
    ySignal = y.';
else
    ySignal = y;
end

coder.internal.errorIf(size(xSignal,1) ~= size(ySignal,1),'signal:dtw:RowMismatch');

if (strcmp(metric,"symmkl") && (~isreal(xSignal) || ~isreal(ySignal) || any(xSignal(:)<0) || any(ySignal(:)<0)))
    coder.internal.error('signal:dtw:MustBeRealPositive');
end

if ~isempty(xSignal) && ~isempty(ySignal)
    if isfinite(r)
        validateattributes(r,{'numeric'},{'integer','positive','scalar','real'},'dtw','MAXSAMP',3)
        C = constrainedCumulativeDistance(xSignal, ySignal, double(r), metric);
        dist = C(size(xSignal,2),size(ySignal,2));
    else
        validateattributes(r,{'numeric'},{'positive','scalar','real'},'dtw','MAXSAMP',3);
        C = unconstrainedCumulativeDistance(xSignal, ySignal, metric);
        dist = C(size(xSignal,2),size(ySignal,2));
    end
    [ix1,iy1] = traceback(C);

else
    if isa(x,'single') || isa(y,'single')
        dist = single(NaN);
    else
        dist = NaN;
    end
    ix1 = zeros(1,0);
    iy1 = zeros(1,0);
end


if nargout==0 && coder.target('MATLAB')
    dtwplot(xSignal, ySignal, ix1, iy1, dist, convertStringsToChars(metric))
else
    if needsTranspose
        ix = ix1';
        iy = iy1';
    else
        ix = ix1;
        iy = iy1;
    end
end
end


%-------------------------------------------------------------------------
function C = unconstrainedCumulativeDistance(x, y, metric)
if isreal(x) && isreal(y)
    C = dtwImpl(x, y, metric);
else
    C = dtwImpl([real(x); imag(x)], [real(y); imag(y)], metric);
end
end

%-------------------------------------------------------------------------
function C = constrainedCumulativeDistance(x, y, r, metric)
m = size(x,2);
n = size(y,2);
if m<n
    if isreal(x) && isreal(y)
        C = dtwImpl(y, x, r, metric)';
    elseif ~isreal(x) && ~isreal(y)
        C = dtwImpl([real(y); imag(y)], [real(x); imag(x)], r, metric)';
    end
elseif isreal(x) && isreal(y)
    C = dtwImpl(x, y, r, metric);
else
    C = dtwImpl([real(x); imag(x)], [real(y); imag(y)], r, metric);
end
end

%-------------------------------------------------------------------------
function [ix_out,iy_out] = traceback(C)
m = size(C,1);
n = size(C,2);

% pre-allocate to the maximum warping path size.
ix = zeros(m+n,1);
iy = zeros(m+n,1);

ix(1) = m;
iy(1) = n;

i = m;
j = n;
k = 1;

while i>1 || j>1
    if j == 1
        i = i-1;
    elseif i == 1
        j = j-1;
    else
        % trace back to the origin, ignoring any NaN value
        % prefer i in a tie between i and j
        cij = C(i-1,j-1);
        ci = C(i-1,j);
        cj = C(i,j-1);
        i = i - (ci<=cj | cij<=cj | cj~=cj);
        j = j - (cj<ci | cij<=ci | ci~=ci);
    end
    k = k+1;
    ix(k) = i;
    iy(k) = j;
end

ix_out = zeros(k,1);
iy_out = zeros(k,1);

for id = 1:k
    ix_out(id) = ix(k-id+1);
    iy_out(id) = iy(k-id+1);
end
end

%-------------------------------------------------------------------------
function C = dtwImpl(x,y,varargin)
%Call dtw Implementation based on target
if coder.target('MATLAB')
    C = dtwmex(x,y,varargin{:});
else
    C = signal.internal.codegenable.dtw.dtwImpl(x,y,varargin{:});
end
end
%-------------------------------------------------------------------------
