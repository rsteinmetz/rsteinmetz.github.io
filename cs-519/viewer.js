/*

viewer.js - v1.0 - public domain - Rick Steinmetz

This file was based on work done by Pavol Klacansky using his dvr.js package. Modifications were made to reduce the size of the code for our purposes.

Based On: 
dvr.js - v2.0 - public domain - Pavol Klacansky

*/

export {container}

const positions = new Float32Array([
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        1.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
        1.0, 0.0, 1.0,
        0.0, 1.0, 1.0,
        1.0, 1.0, 1.0,
])

const indices = new Uint16Array([
        /* bottom */
        0, 2, 1,
        2, 3, 1,
        /* top */
        4, 5, 6,
        6, 5, 7,
        /* left */
        2, 0, 6,
        6, 0, 4,
        /* right */
        1, 3, 5,
        5, 3, 7,
        /* back */
        3, 2, 7,
        7, 2, 6,
        /* front */
        0, 1, 4,
        4, 1, 5,
])

const indicesBox = new Uint16Array([
        0, 1,
        0, 2,
        1, 3,
        2, 3,
        0, 4,
        1, 5,
        2, 6,
        3, 7,
        4, 5,
        4, 6,
        5, 7,
        6, 7,

])

const vertSrcBox = `
#version 300 es

layout (location = 0) in vec4 pos;
layout (std140) uniform Matrices {
        mat4 model;
        mat4 view;
        mat4 projection;
} matrices;

void
main(void)
{
        gl_Position = matrices.projection*matrices.view*matrices.model*pos;
}
`.trim()


const fragSrcBox = `
#version 300 es

precision mediump float;

layout(location = 0) out vec4 color;

void
main(void)
{
        color = vec4(1.0, 1.0, 1.0, 1.0);
}
`.trim()

function mkVertSrc() {
        return `
#version 300 es

layout(location = 0) in vec4 pos;
layout(std140) uniform Matrices {
        mat4 model;
        mat4 view;
        mat4 projection;
} matrices;

out vec3 v_pos;
out vec3 v_world_pos;
out vec3 eye;
out mat4 to_world;
out mat3 to_worldn;
out float zScaling;

void
main(void)
{
        /* needed to scale tFar in fragment shader as we have to use the insane GL projection matrix */
        zScaling = matrices.projection[2][2];

        /* TODO: precompute inverse */
        to_world = matrices.view*matrices.model;
        mat4 inv = inverse(to_world);
        to_worldn = transpose(mat3(inv));
        eye = (inv*vec4(0.0, 0.0, 0.0, 1.0)).xyz;
        vec4 position = pos;
        v_pos = position.xyz;
        v_world_pos = (matrices.view*matrices.model*position).xyz;
        gl_Position = matrices.projection*matrices.view*matrices.model*position;
}
`.trim()
}

function mkFragSrc() {
        return `
#version 300 es

#define SURFACE 0
#define VOLUME 1
#define METHOD ${'SURFACE'}

precision mediump float;

uniform highp sampler3D volume_sampler;
uniform mediump sampler2D transfer_function_sampler;
uniform highp sampler2D depth_sampler;

uniform float isovalue;

in vec3 v_pos;
in vec3 v_world_pos;
in vec3 eye;
in mat4 to_world;
in mat3 to_worldn;
in float zScaling;

layout(location = 0) out vec4 color;


/* from internet (unknown original source) */
float
rand(vec2 co)
{
        return fract(sin(dot(co.xy, vec2(12.9898, 78.233)))*43758.5453);
}


/* central difference */
vec3
gradient(in sampler3D s, vec3 p, float dt)
{
        vec2 e = vec2(dt, 0.0);

        return vec3(texture(s, p - e.xyy).r - texture(s, p + e.xyy).r,
                texture(s, p - e.yxy).r - texture(s, p + e.yxy).r,
                texture(s, p - e.yyx).r - texture(s, p + e.yyx).r);
}


float
linear_to_srgb(float linear)
{
        if (linear <= 0.0031308)
                return 12.92*linear;
        else
                return (1.0 + 0.055)*pow(linear, 1.0/2.4) - 0.055;
}


void
main(void)
{
        const vec3 light_pos = vec3(1.0, 1.0, 1.0);
        vec3 o = eye;
        vec3 d = normalize(v_pos - o);

        /* intersect aabb */
        vec3 near = min(-o/d, (vec3(1.0) - o)/d);
        vec3 far = max(-o/d, (vec3(1.0) - o)/d);
        float tnear = max(near.x, max(near.y, near.z));
        float tfar  = min(far.x, min(far.y, far.z));
        if (tnear > tfar)
                discard;

        /* stop at geometry if there is any (do ratio of z coordinate depth and z coordinate of current fragment) */
        float depth = texelFetch(depth_sampler, ivec2(gl_FragCoord.xy), 0).r;
        tfar *= min((zScaling + gl_FragCoord.z)/(zScaling + depth), 1.0);

        ivec3 size = textureSize(volume_sampler, 0);
        int max_size = max(size.x, max(size.y, size.z));

        /* compute step size (3D DDA) */
        vec3 cell_size = 1.0/vec3(size);
        vec3 dts = cell_size/abs(d);
        float dt = min(dts.x, min(dts.y, dts.z));

        /* create safe bubble close to head if it is inside the volume */
        const float head_bubble_radius = 0.2;
        if (tnear < head_bubble_radius)
                tnear = floor((head_bubble_radius - tnear)/dt)*dt + tnear;

        color = vec4(0.0, 0.0, 0.0, 0.0);
        float prev_value = 0.0;
        float t = tnear + dt*rand(gl_FragCoord.xy);

        for (int i = 0; i < max_size; ++i) {
                if (t >= tfar)
                        break;

                vec3 p = o + t*d;
                float value = texture(volume_sampler, p).r;
                if (sign(value - isovalue) != sign(prev_value - isovalue)) {

                        // set the base or most inner color to be white to represent bone
                        color = vec4(1.0, 1.0, 1.0, 1.0); 

                        // as the isovalue (value) changes, we need to adjust the 
                        if (value < 0.5) {
                            // pink color
                            color = vec4(0.78, 0.13, 0.13, 1.0); /* 0.9, 0.4, 0.4 in sRGB space */
                        }

                        if (value <= 0.15) {
                            color = vec4(0.80, 0.76078, 0.59608, 1.0); /* 0.9, 0.4, 0.4 in sRGB space */
                        }

                        /* linear approximation of intersection point */
                        vec3 prev_p = p - dt*d;
                        float a = (isovalue - prev_value)/(value - prev_value);
                        vec3 inter_p = (1.0 - a)*(p - dt*d) + a*p;
                        /* TODO: sample at different dt for each axis to avoid having undo scaling */
                        vec3 nn = gradient(volume_sampler, inter_p, dt);

                        /* TODO: can we optimize somehow? */
                        vec3 world_p = (to_world*vec4(inter_p, 1.0)).xyz;
                        vec3 n = normalize(to_worldn*nn);
                        vec3 light_dir = normalize(light_pos - world_p);
                        vec3 h = normalize(light_dir - world_p); /* eye is at origin */
                        const float ambient = 0.2;
                        float diffuse = 0.6*clamp(dot(light_dir, n), 0.0, 1.0);
                        float specular = 0.2*pow(clamp(dot(h, n), 0.0, 1.0), 100.0);
                        float distance = length(world_p); /* eye is at origin */
                        color.rgb = color.rgb*(ambient + (diffuse + specular)/distance);
                        break;
                }
                prev_value = value;
                t += dt;
        }
        if (color.a == 0.0)
                discard;

        color.rgb = vec3(linear_to_srgb(color.r), linear_to_srgb(color.g), linear_to_srgb(color.b));
}
`.trim()
}

function vec2(x, y) {
    return {x, y}
}

function vec3(x, y, z) {
    return {x, y, z}
}

function vec3_dot(u, v) {
    return u.x*v.x + u.y*v.y + u.z*v.z
}

function vec3_cross(u, v) {
    return vec3(u.y*v.z - u.z*v.y,
                u.z*v.x - u.x*v.z,
                u.x*v.y - u.y*v.x)
}

function mat4(...ms) {
    return ms
}


function mat4_scale(x, y, z) {
    return mat4(x, 0, 0, 0,
                0, y, 0, 0,
                0, 0, z, 0,
                0, 0, 0, 1)
}


function mat4_translate(x, y, z) {
    return mat4(1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0,
                x, y, z, 1)
}


function mat4_mul(m1, m2){
    const m = []
    for (let i = 0; i < 4; ++i)
    for (let j = 0; j < 4; ++j) {
            m[4*j + i] = 0
            for (let k = 0; k < 4; ++k)
                    m[4*j + i] += m1[4*k + i]*m2[4*j + k]
    }
    return m
}


function quat(w, x, y, z) {
    return {w, x, y, z}
}

function quat_mul(q0, q1) {
    return quat(q0.w*q1.w - q0.x*q1.x - q0.y*q1.y - q0.z*q1.z,
                q0.w*q1.x + q1.w*q0.x + q0.y*q1.z - q0.z*q1.y,
                q0.w*q1.y + q1.w*q0.y - q0.x*q1.z + q0.z*q1.x,
                q0.w*q1.z + q1.w*q0.z + q0.x*q1.y - q0.y*q1.x)
}

function quat_to_mat4(q) {
    return mat4(1.0 - 2.0*q.y*q.y - 2.0*q.z*q.z, 2.0*q.x*q.y + 2.0*q.w*q.z, 2.0*q.x*q.z - 2.0*q.w*q.y, 0.0,
                2.0*q.x*q.y - 2.0*q.w*q.z, 1.0 - 2.0*q.x*q.x - 2.0*q.z*q.z, 2.0*q.y*q.z + 2.0*q.w*q.x, 0.0,
                2.0*q.x*q.z + 2.0*q.w*q.y, 2.0*q.y*q.z - 2.0*q.w*q.x, 1.0 - 2.0*q.x*q.x - 2.0*q.y*q.y, 0.0,
                0.0, 0.0, 0.0, 1.0
    )
}

function arcball_screen_to_sphere(circle, screen_x, screen_y) {
    const x = (screen_x - circle.center.x)/circle.radius
    const y = -(screen_y - circle.center.y)/circle.radius
    const r = x*x + y*y

    if (r > 1.0) {
            const s = 1.0/Math.sqrt(r)
            return vec3(s*x, s*y, 0.0)
    } else
            return vec3(x, y, Math.sqrt(1.0 - r))
}

function arcball_quat(start_point, end_point) {
    const axis  = vec3_cross(start_point, end_point)
    const angle = vec3_dot(start_point, end_point)
    return quat(angle, axis.x, axis.y, axis.z)
}

const to_half = (() => {
        const floatView = new Float32Array(1);
        const int32View = new Int32Array(floatView.buffer);

        return val => {
                floatView[0] = val
                const x = int32View[0]

                let bits = (x >> 16) & 0x8000 /* Get the sign */
                let m = (x >> 12) & 0x07ff /* Keep one extra bit for rounding */
                const e = (x >> 23) & 0xff /* Using int is faster here */

                /* If zero, or denormal, or exponent underflows too much for a denormal
                 * half, return signed zero. */
                if (e < 103) {
                        return bits
                }

                /* If NaN, return NaN. If Inf or exponent overflow, return Inf. */
                if (e > 142) {
                        bits |= 0x7c00
                        /* If exponent was 0xff and one mantissa bit was set, it means NaN,
                         * not Inf, so make sure we set one mantissa bit too. */
                        bits |= ((e == 255) ? 0 : 1) && (x & 0x007fffff)
                        return bits
                }

                /* If exponent underflows but not too much, return a denormal */
                if (e < 113) {
                        m |= 0x0800
                        /* Extra rounding may overflow and set mantissa to 0 and exponent
                        * to 1, which is OK. */
                        bits |= (m >> (114 - e)) + ((m >> (113 - e)) & 1)
                        return bits
                }

                bits |= ((e - 112) << 10) | (m >> 1)
                /* Extra rounding. An overflow will set mantissa to 0 and increment
                * the exponent, which is OK. */
                bits += m & 1
                return bits
        }
})()

function container(canvas) {
        const viewer = {}

        const width = canvas.width
        const height = canvas.height

        const arcball_circle = {
                center: vec2(width/2, height/2),
                radius: Math.min(width/2, height/2),
        }

        let viewDistance = 1.5 

        const near_plane = 0.01
        const far_plane = 1000.0
        const matrices = {
                model:      mat4(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -0.5, -0.5, -0.5, 1.0),
                view:       mat4(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -viewDistance, 1.0),
                projection: mat4(1.0, 0.0, 0.0, 0.0,
                                 0.0, 1.0, 0.0, 0.0,
                                 0.0, 0.0, -(far_plane + near_plane)/(far_plane - near_plane), -1.0,
                                 0.0, 0.0, -2.0*far_plane*near_plane/(far_plane - near_plane), 0.0),
        }

        let p0
        let q = quat(1.0, 0.0, 0.0, 0.0)
        let q_down

        canvas.addEventListener('mousedown', e => {
                if (!(e.buttons & 1))
                        return

                q_down = q
                const rect = canvas.getBoundingClientRect()
                p0 = arcball_screen_to_sphere(arcball_circle, e.clientX - rect.left, e.clientY - rect.top)
        })
        canvas.addEventListener('mousemove', e => {
                if (!(e.buttons & 1))
                        return

                const rect = canvas.getBoundingClientRect()
                const p1 = arcball_screen_to_sphere(arcball_circle, e.clientX - rect.left, e.clientY - rect.top)
                const q_move = arcball_quat(p0, p1)

                q = quat_mul(q_move, q_down)

                matrices.view = quat_to_mat4(q)
                /* translate camera */
                /* TODO: bit error prone as we have it in another place (`present` function) */
                matrices.view[14] = -viewDistance;

                render(new Float32Array(matrices.view), new Float32Array(matrices.projection), fbos, 0, 0, canvas.width, canvas.height)
        })
        canvas.addEventListener('wheel', e => {
                e.preventDefault()

                viewDistance = Math.max(1, viewDistance + 0.1*Math.sign(e.deltaY))

                matrices.view[14] = -viewDistance;

                render(new Float32Array(matrices.view), new Float32Array(matrices.projection), fbos, 0, 0, canvas.width, canvas.height)
        })


        const gl = canvas.getContext('webgl2', {alpha: false, antialias: false, depth: false, stencil: false})
        if (!gl) {
                console.log('WebGL: version 2 not available')
                alert('WebGL 2 is not available')
        }

        let context = gl.getExtension('WEBGL_lose_context')

        let currentDataSet
        let isovalue
        let transferFunction

        /* handle context loss and restore */
        canvas.addEventListener('webglcontextlost', e => {
                console.log('WebGL: lost context')
                e.preventDefault()
                //setTimeout(() => context.restoreContext(), 0)
        })
        canvas.addEventListener('webglcontextrestored', () => {
                console.log('WebGL: restored context')
                init()

                if (currentDataSet) {
                        viewer.uploadData(currentDataSet['typedArray'], currentDataSet['width'], currentDataSet['height'], currentDataSet['depth'], currentDataSet['boxWidth'], currentDataSet['boxHeight'], currentDataSet['boxDepth'])
                }
                
                viewer.isovalue(isovalue)

                render(new Float32Array(matrices.view), new Float32Array(matrices.projection), fbos, 0, 0, canvas.width, canvas.height)
        })


        let volumeTex, transferFunctionTex, vbo, ebo, program, eboBox, programBox, ubo, fbos, volumeSampler, transferFunctionSampler, depthSampler

        function init() {
                /* necessary for linear filtering of float textures */
                if (!gl.getExtension('OES_texture_float_linear'))
                        console.log('WebGL: no linear filtering for float textures')

                gl.enable(gl.DEPTH_TEST)
                gl.enable(gl.CULL_FACE)
                gl.cullFace(gl.FRONT)

                program = createProgram(gl, mkVertSrc(), mkFragSrc())
                gl.uniformBlockBinding(program, gl.getUniformBlockIndex(program, 'Matrices'), 0)

                programBox = createProgram(gl, vertSrcBox, fragSrcBox)
                gl.uniformBlockBinding(programBox, gl.getUniformBlockIndex(programBox, 'Matrices'), 0)
                
                vbo = gl.createBuffer()
                gl.bindBuffer(gl.ARRAY_BUFFER, vbo)
                gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW)

                ebo = gl.createBuffer()
                gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ebo)
                gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indices, gl.STATIC_DRAW)

                /* dummy vao */
                const vao = gl.createVertexArray()
                gl.bindVertexArray(vao)

                /* bounding box wireframe */
                eboBox = gl.createBuffer()
                gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, eboBox)
                gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indicesBox, gl.STATIC_DRAW)

                fbos = createFbos(gl, width, height)

                transferFunctionSampler = gl.createSampler()
                gl.samplerParameteri(transferFunctionSampler, gl.TEXTURE_MIN_FILTER, gl.LINEAR)
                gl.samplerParameteri(transferFunctionSampler, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
                gl.samplerParameteri(transferFunctionSampler, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)

                depthSampler = gl.createSampler()
                gl.samplerParameteri(depthSampler, gl.TEXTURE_MAG_FILTER, gl.NEAREST)
                gl.samplerParameteri(depthSampler, gl.TEXTURE_MIN_FILTER, gl.NEAREST)

                ubo = gl.createBuffer()
                gl.bindBuffer(gl.UNIFORM_BUFFER, ubo)
                gl.bufferData(gl.UNIFORM_BUFFER, new Float32Array(matrices.model.concat(matrices.view, matrices.projection)), gl.DYNAMIC_DRAW)

                gl.useProgram(program)
                gl.uniform1i(gl.getUniformLocation(program, 'transfer_function_sampler'), 1)
                gl.uniform1i(gl.getUniformLocation(program, 'depth_sampler'), 2)

                volumeTex = undefined
                transferFunctionTex = undefined
        }



        /* inline as it is more convenient */
        function render(view_matrix, projection_matrix, fbos, x, y, width, height) {
                gl.viewport(0, 0, fbos.width, fbos.height)

                gl.bindBuffer(gl.UNIFORM_BUFFER, ubo)
                gl.bufferSubData(gl.UNIFORM_BUFFER, 16*4, view_matrix)
                gl.bufferSubData(gl.UNIFORM_BUFFER, 32*4, projection_matrix)

                gl.bindFramebuffer(gl.FRAMEBUFFER, fbos.fbo)
                // gl.clearBufferfv(gl.COLOR, fbos.fbo, [255/255, 246/255, 213/255, 1.0])
                // set background to black
                gl.clearBufferfv(gl.COLOR, fbos.fbo, [0/255, 0/255, 0/255, 1.0])
                //gl.clearBufferfv(gl.DEPTH, fbos.fbo, [1.0])
                gl.clearBufferfv(gl.DEPTH, fbos.fbo, [1.0])


                gl.bindBufferBase(gl.UNIFORM_BUFFER, 0, ubo)
                
                gl.useProgram(programBox)
                gl.bindBuffer(gl.ARRAY_BUFFER, vbo)
                gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, eboBox)
                gl.enableVertexAttribArray(0)
                gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0)
                gl.drawElements(gl.LINES, indicesBox.length, gl.UNSIGNED_SHORT, 0)


                /* volume/surface rendering goes last as it needs to read depth from previous passes */
                gl.disable(gl.DEPTH_TEST) 
                gl.bindFramebuffer(gl.FRAMEBUFFER, fbos.volumeFbo)

                gl.activeTexture(gl.TEXTURE0)
                gl.bindTexture(gl.TEXTURE_3D, volumeTex)
                gl.bindSampler(0, volumeSampler)

                /* depth buffer texture */
                gl.activeTexture(gl.TEXTURE2)
                gl.bindTexture(gl.TEXTURE_2D, fbos.depthTexture)
                gl.bindSampler(2, depthSampler)

                gl.useProgram(program)
                gl.bindBuffer(gl.ARRAY_BUFFER, vbo)
                gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ebo)
                gl.enableVertexAttribArray(0)
                gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0)
                gl.drawElements(gl.TRIANGLES, indices.length, gl.UNSIGNED_SHORT, 0)

                gl.enable(gl.DEPTH_TEST)

                /* copy to canvas framebuffer */
                gl.bindFramebuffer(gl.READ_FRAMEBUFFER, fbos.volumeFbo)
                gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, null)
                gl.blitFramebuffer(0, 0, fbos.width, fbos.height,
                                   x, y, x + width, y + height,
                                   gl.COLOR_BUFFER_BIT, gl.LINEAR)
        }


        /* 8 and 16 bit integers are converted to float16 */
        viewer.uploadData = (typedArray, width, height, depth, boxWidth, boxHeight, boxDepth) => {
                currentDataSet = {
                        'typedArray': typedArray,
                        'width': width,
                        'height': height,
                        'depth': depth,
                        'boxWidth': boxWidth,
                        'boxHeight': boxHeight,
                        'boxDepth': boxDepth,
                }

                /* TODO: dealloc or if same size then reuse */
                volumeTex = gl.createTexture()
                gl.bindTexture(gl.TEXTURE_3D, volumeTex)

                /* update isovalue slider and shader uniform */
                const extent = [typedArray[0], typedArray[0]]
                for (let i = 0; i < typedArray.length; i++) {
                        extent[0] = Math.min(typedArray[i], extent[0])
                        extent[1] = Math.max(typedArray[i], extent[1])
                }

                gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1)
                if (typedArray instanceof Int8Array || typedArray instanceof Int16Array || typedArray instanceof Uint8Array || typedArray instanceof Uint16Array) {
                        const converted = new Uint16Array(typedArray.length)
                        for (let i = 0; i < typedArray.length; i++) {
                                converted[i] = to_half((typedArray[i] - extent[0])/(extent[1] - extent[0]))
                        }
                        gl.texStorage3D(gl.TEXTURE_3D, 1, gl.R16F, width, height, depth)
                        gl.texSubImage3D(gl.TEXTURE_3D, 0,
                                         0, 0, 0,
                                         width, height, depth,
                                         gl.RED, gl.HALF_FLOAT, converted)

                } else if (typedArray instanceof Float32Array) {
                        const converted = new Float32Array(typedArray.length)
                        for (let i = 0; i < typedArray.length; i++) {
                                converted[i] = (typedArray[i] - extent[0])/(extent[1] - extent[0])
                        }
                        gl.texStorage3D(gl.TEXTURE_3D, 1, gl.R32F, width, height, depth)
                        gl.texSubImage3D(gl.TEXTURE_3D, 0,
                                         0, 0, 0,
                                         width, height, depth,
                                         gl.RED, gl.FLOAT, converted)
                } else if (typedArray instanceof Float64Array) {
                        const converted = new Float32Array(typedArray.length)
                        for (let i = 0; i < typedArray.length; i++) {
                                converted[i] = (typedArray[i] - extent[0])/(extent[1] - extent[0])
                        }
                        gl.texStorage3D(gl.TEXTURE_3D, 1, gl.R32F, width, height, depth)
                        gl.texSubImage3D(gl.TEXTURE_3D, 0,
                                         0, 0, 0,
                                         width, height, depth,
                                         gl.RED, gl.FLOAT, converted)
                } else {
                        console.log('Unsupported array type')
                }


                const max = Math.max(boxWidth, boxHeight, boxDepth)
                matrices.model = mat4_mul(mat4_scale(boxWidth/max, boxHeight/max, boxDepth/max), mat4_translate(-0.5, -0.5, -0.5))
                gl.bindBuffer(gl.UNIFORM_BUFFER, ubo)
                gl.bufferSubData(gl.UNIFORM_BUFFER, 0, new Float32Array(matrices.model))

                return viewer
        }

        /* isovalue must be within range [0,1] */
        viewer.isovalue = (value) => {
            isovalue = value
            console.log(isovalue)

            gl.useProgram(program)
            gl.uniform1f(gl.getUniformLocation(program, 'isovalue'), value)

            render(new Float32Array(matrices.view), new Float32Array(matrices.projection), fbos, 0, 0, canvas.width, canvas.height)
            return viewer
        }

        /* setup scene */
        {
                viewDistance = 1.5
                q = quat(1.0, 0.0, 0.0, 0.0)
                matrices.view       = mat4(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -viewDistance, 1.0)
                matrices.projection = mat4(1.0, 0.0, 0.0, 0.0,
                                 0.0, 1.0, 0.0, 0.0,
                                 0.0, 0.0, -(far_plane + near_plane)/(far_plane - near_plane), -1.0,
                                 0.0, 0.0, -2.0*far_plane*near_plane/(far_plane - near_plane), 0.0)

                canvas.width = width
                canvas.height = height
        }

        init()

        return viewer
}

function createProgram(gl, vertSrc, fragSrc) {
        /* create program from two shaders */
        const vertexShader = gl.createShader(gl.VERTEX_SHADER)
        gl.shaderSource(vertexShader, vertSrc)
        gl.compileShader(vertexShader)
        if (gl.getShaderInfoLog(vertexShader))
                console.log('WebGL:', gl.getShaderInfoLog(vertexShader))

        const fragmentShader = gl.createShader(gl.FRAGMENT_SHADER)
        gl.shaderSource(fragmentShader, fragSrc)
        gl.compileShader(fragmentShader)
        if (gl.getShaderInfoLog(fragmentShader))
                console.log('WebGL:', gl.getShaderInfoLog(fragmentShader))

        const program = gl.createProgram()
        gl.attachShader(program, vertexShader)
        gl.attachShader(program, fragmentShader)
        gl.linkProgram(program)

        gl.deleteShader(vertexShader)
        gl.deleteShader(fragmentShader)

        return program
}


function createFbos(gl, width, height) {
        const colorRenderbuffer = gl.createRenderbuffer()
        gl.bindRenderbuffer(gl.RENDERBUFFER, colorRenderbuffer)
        gl.renderbufferStorage(gl.RENDERBUFFER, gl.SRGB8_ALPHA8, width, height)

        const depthTexture = gl.createTexture()
        gl.bindTexture(gl.TEXTURE_2D, depthTexture)
        gl.texStorage2D(gl.TEXTURE_2D, 1, gl.DEPTH_COMPONENT32F, width, height)

        /* framebuffer for rendering geometry */
        const fbo = gl.createFramebuffer()
        gl.bindFramebuffer(gl.FRAMEBUFFER, fbo)
        gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.RENDERBUFFER, colorRenderbuffer)
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.TEXTURE_2D, depthTexture, 0)

        /* framebuffer for rendering volume/surface */
        const volumeFbo = gl.createFramebuffer()
        gl.bindFramebuffer(gl.FRAMEBUFFER, volumeFbo)
        gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.RENDERBUFFER, colorRenderbuffer)

        return {fbo, volumeFbo, depthTexture, width, height}
}
